"""A predictive-coding hierarchy that learns with local rules only.

Each level holds activities ``r`` that try to explain the level below. Generative
weights ``W`` render a prediction of that input; the difference is the prediction
error ``e``. Activities settle by feedforward drive of the error through separate
weights ``B``, minus a threshold (sparse coding, Olshausen & Field 1996; Rao &
Ballard 1999). After settling, every connection changes by the product of the
activities at its two ends:

* ``W`` and ``B`` both by error × activity (Kolen-Pollack learning: ``B`` never
  reads ``W``, yet the shared local update and normalisation make them converge),
* thresholds by homeostasis toward a target firing rate,
* a temporal predictor ``V`` by the normalised delta rule: (what happened − what it
  linearly predicted) × what came before, divided by the presynaptic energy that a
  cell can sense in its own input.

No automatic differentiation is used anywhere.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.nn.grad import conv2d_weight


# A newborn's feedback weights start unrelated to its generative weights, which
# makes the settling operator weak and the computed step large; uncapped, every
# cell saturates at once and the shared Hebbian update makes all cells identical.
MAX_SETTLING_STEP = 0.1
# No single frame may move a cell's weights by more than this fraction of their
# (unit) length: plasticity is bounded per cell, locally.
MAX_WEIGHT_CHANGE = 0.05


@dataclass(frozen=True)
class LevelConfig:
    cells: int = 32
    kernel: int = 8
    stride: int = 2
    padding: int = 3
    sparsity: float = 0.05
    learning_rate: float = 10.0
    target_rate: float = 0.05
    temporal_kernel: int = 9
    temporal_rate: float = 0.5


@dataclass
class StepResult:
    activities: list[torch.Tensor]
    errors: list[torch.Tensor]
    code_error: float
    persistence_error: float
    predicted_next: torch.Tensor
    # The cortex's own "nothing changes" expectation: its current representation,
    # rendered unchanged. Comparing against it isolates what temporal learning added.
    stillness_error: float = float("nan")


def _normalise(weights: torch.Tensor) -> torch.Tensor:
    """Synaptic scaling: every cell's incoming weights keep unit length."""

    return weights / weights.flatten(1).norm(dim=1).clamp_min(1e-8)[:, None, None, None]


class Level:
    def __init__(self, in_channels: int, config: LevelConfig, device, generator: torch.Generator):
        self.config = config
        shape = (config.cells, in_channels, config.kernel, config.kernel)
        self.W = _normalise(torch.randn(shape, generator=generator).to(device))
        # Feedback starts independent of W: nothing is ever copied between them.
        self.B = _normalise(torch.randn(shape, generator=generator).to(device))
        self.theta = torch.full((1, config.cells, 1, 1), config.sparsity, device=device)
        # Temporal predictor starts as persistence: each cell expects itself to continue.
        k = config.temporal_kernel
        self.V = torch.zeros(config.cells, config.cells, k, k, device=device)
        self.V[torch.arange(config.cells), torch.arange(config.cells), k // 2, k // 2] = 1.0
        self.step_size = 0.05

    def render(self, activity: torch.Tensor) -> torch.Tensor:
        c = self.config
        return F.conv_transpose2d(activity, self.W, stride=c.stride, padding=c.padding)

    def drive(self, error: torch.Tensor) -> torch.Tensor:
        c = self.config
        return F.conv2d(error, self.B, stride=c.stride, padding=c.padding)

    def forecast(self, activity: torch.Tensor) -> torch.Tensor:
        return F.conv2d(activity, self.V, padding=self.config.temporal_kernel // 2).clamp(0.0, 1.0)

    def settle(self, x: torch.Tensor, activity: torch.Tensor, iterations: int) -> torch.Tensor:
        for _ in range(iterations):
            error = x - self.render(activity)
            activity = (activity + self.step_size * (self.drive(error) - self.theta)).clamp(
                0.0, 1.0
            )
        return activity

    def stability_limit(self, shape: tuple[int, ...], iterations: int = 20) -> float:
        """Largest eigenvalue of the settling operator, by power iteration."""

        vector = torch.rand(1, self.config.cells, *shape[-2:], device=self.W.device)
        norm = torch.tensor(1.0)
        for _ in range(iterations):
            vector = self.drive(self.render(vector))
            norm = vector.norm()
            vector = vector / norm.clamp_min(1e-8)
        return float(norm)

    def learn(self, x, activity, error, previous) -> None:
        c = self.config
        positions = activity.shape[0] * activity.shape[2] * activity.shape[3]
        hebb = conv2d_weight(error, self.W.shape, activity, stride=c.stride, padding=c.padding)
        update = c.learning_rate * hebb / positions
        size = update.flatten(1).norm(dim=1).clamp_min(1e-12)
        update = update * (MAX_WEIGHT_CHANGE / size).clamp(max=1.0)[:, None, None, None]
        self.W = _normalise(self.W + update)
        self.B = _normalise(self.B + update)
        rate = (activity > 0).float().mean((0, 2, 3), keepdim=True)
        adjust = torch.exp(0.002 * (rate / c.target_rate - 1.0).clamp(-1.0, 1.0))
        self.theta = (self.theta * adjust).clamp(c.sparsity / 10, c.sparsity * 10)
        if previous is not None:
            # Learn against the linear prediction: the clamp is for output only, and a
            # rule that learns against a saturated output has no fixed point.
            linear = F.conv2d(previous, self.V, padding=c.temporal_kernel // 2)
            delta = conv2d_weight(
                previous, self.V.shape, activity - linear, padding=c.temporal_kernel // 2
            )
            energy = previous.square().mean() * c.cells * c.temporal_kernel**2
            self.V = self.V + c.temporal_rate * delta / positions / energy.clamp_min(1e-6)


class Hierarchy:
    def __init__(self, in_channels: int, levels: list[LevelConfig], device="cpu", seed: int = 0):
        self.in_channels = in_channels
        self.configs = list(levels)
        self.device = torch.device(device)
        self.seed = seed
        self.reset_learning(seed)

    def reset_learning(self, seed: int | None = None) -> None:
        """Back to an unlearned newborn: fresh random weights, no experience."""

        generator = torch.Generator().manual_seed(self.seed if seed is None else seed)
        channels, self.levels = self.in_channels, []
        for config in self.configs:
            self.levels.append(Level(channels, config, self.device, generator))
            channels = config.cells
        self.age_frames = 0
        self.reset_state()

    def reset_state(self) -> None:
        self._activity: torch.Tensor | None = None
        self._predicted_activity: torch.Tensor | None = None
        self._predicted_code: torch.Tensor | None = None
        self._previous_code: torch.Tensor | None = None
        self._still_code: torch.Tensor | None = None

    @torch.no_grad()
    def step(self, code: torch.Tensor, learn: bool = True, iterations: int = 20) -> StepResult:
        level = self.levels[0]
        code = code.to(self.device)
        if self.age_frames % 100 == 0 or level.step_size <= 0:
            limit = max(level.stability_limit(code.shape), 1e-6)
            level.step_size = min(0.9 / limit, MAX_SETTLING_STEP)
        code_error = persistence_error = stillness_error = float("nan")
        if self._predicted_code is not None and self._predicted_code.shape == code.shape:
            code_error = float((self._predicted_code - code).square().mean())
            persistence_error = float((self._previous_code - code).square().mean())
            stillness_error = float((self._still_code - code).square().mean())
        # Warm start from what the level expected to see.
        start = self._predicted_activity
        if start is None or start.shape[0] != code.shape[0]:
            start = torch.zeros(
                code.shape[0],
                level.config.cells,
                code.shape[2] // level.config.stride,
                code.shape[3] // level.config.stride,
                device=self.device,
            )
        activity = level.settle(code, start, iterations)
        error = code - level.render(activity)
        if learn:
            level.learn(code, activity, error, self._activity)
            self.age_frames += 1
        self._activity = activity
        self._predicted_activity = level.forecast(activity)
        self._predicted_code = level.render(self._predicted_activity)
        self._still_code = level.render(activity)
        self._previous_code = code
        return StepResult(
            activities=[activity],
            errors=[error],
            code_error=code_error,
            persistence_error=persistence_error,
            predicted_next=self._predicted_code,
            stillness_error=stillness_error,
        )

    def predict_next_code(self) -> torch.Tensor | None:
        return self._predicted_code

    def architecture(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "levels": [dataclasses.asdict(config) for config in self.configs],
        }

    def state_dict(self) -> dict:
        weights = {}
        for index, level in enumerate(self.levels):
            for name in ("W", "B", "V", "theta"):
                weights[f"{index}.{name}"] = getattr(level, name).detach().cpu().clone()
        return {
            "architecture": self.architecture(),
            "weights": weights,
            "age_frames": self.age_frames,
        }

    def load_state_dict(self, state: dict) -> None:
        if state["architecture"] != self.architecture():
            raise ValueError(
                "checkpoint architecture does not match this network: "
                f"{state['architecture']} vs {self.architecture()}"
            )
        for index, level in enumerate(self.levels):
            for name in ("W", "B", "V", "theta"):
                setattr(level, name, state["weights"][f"{index}.{name}"].to(self.device))
        self.age_frames = int(state["age_frames"])
        self.reset_state()
