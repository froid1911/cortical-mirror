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
# How strongly a level's activity is pulled toward the prediction from above.
TOP_DOWN_WEIGHT = 0.5


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

        # A fixed starting vector keeps step sizes, and so whole runs, reproducible.
        generator = torch.Generator().manual_seed(0)
        vector = torch.rand(1, self.config.cells, *shape[-2:], generator=generator).to(
            self.W.device
        )
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
        self._activities: list[torch.Tensor] | None = None
        self._predicted_activities: list[torch.Tensor] | None = None
        self._predicted_code: torch.Tensor | None = None
        self._previous_code: torch.Tensor | None = None
        self._still_code: torch.Tensor | None = None

    def _refresh_step_sizes(self, code: torch.Tensor) -> None:
        shape = code.shape
        for level in self.levels:
            limit = max(level.stability_limit(shape), 1e-6)
            level.step_size = min(0.9 / limit, MAX_SETTLING_STEP)
            c = level.config
            shape = (shape[0], c.cells, shape[2] // c.stride, shape[3] // c.stride)

    def _zeros(self, code: torch.Tensor) -> list[torch.Tensor]:
        activities, (height, width) = [], code.shape[-2:]
        for level in self.levels:
            c = level.config
            height, width = height // c.stride, width // c.stride
            activities.append(
                torch.zeros(code.shape[0], c.cells, height, width, device=self.device)
            )
        return activities

    def _errors(self, code: torch.Tensor, activities: list[torch.Tensor]) -> list[torch.Tensor]:
        inputs = [code, *activities[:-1]]
        return [
            x - level.render(a) for x, level, a in zip(inputs, self.levels, activities, strict=True)
        ]

    def _settle(self, code, activities, iterations: int, feedback: bool):
        """Every level is driven by the error it must explain and, unless it is the top,
        pulled toward the prediction the level above makes of it (Rao & Ballard)."""

        for _ in range(iterations):
            errors = self._errors(code, activities)
            updated = []
            for index, (level, activity) in enumerate(zip(self.levels, activities, strict=True)):
                drive = level.drive(errors[index]) - level.theta
                if feedback and index + 1 < len(self.levels):
                    drive = drive - TOP_DOWN_WEIGHT * errors[index + 1]
                updated.append((activity + level.step_size * drive).clamp(0.0, 1.0))
            activities = updated
        return activities, self._errors(code, activities)

    @torch.no_grad()
    def settle(
        self, code: torch.Tensor, iterations: int = 30, feedback: bool = True
    ) -> list[torch.Tensor]:
        """Frozen settle from rest, for measurements: no learning, no state change."""

        code = code.to(self.device)
        saved = [level.step_size for level in self.levels]
        self._refresh_step_sizes(code)
        activities, _ = self._settle(code, self._zeros(code), iterations, feedback)
        for level, step in zip(self.levels, saved, strict=True):
            level.step_size = step
        return activities

    @torch.no_grad()
    def step(self, code: torch.Tensor, learn: bool = True, iterations: int = 20) -> StepResult:
        code = code.to(self.device)
        if self.age_frames % 100 == 0 or self.levels[0].step_size <= 0:
            self._refresh_step_sizes(code)
        code_error = persistence_error = stillness_error = float("nan")
        if self._predicted_code is not None and self._predicted_code.shape == code.shape:
            code_error = float((self._predicted_code - code).square().mean())
            persistence_error = float((self._previous_code - code).square().mean())
            stillness_error = float((self._still_code - code).square().mean())
        # Warm start from what each level expected to see.
        starts = self._predicted_activities
        if starts is None or starts[0].shape[0] != code.shape[0]:
            starts = self._zeros(code)
        activities, errors = self._settle(code, starts, iterations, feedback=True)
        if learn:
            inputs = [code, *activities[:-1]]
            previous = self._activities or [None] * len(self.levels)
            for level, x, a, e, p in zip(
                self.levels, inputs, activities, errors, previous, strict=True
            ):
                level.learn(x, a, e, p)
            self.age_frames += 1
        self._activities = activities
        self._predicted_activities = [
            level.forecast(a) for level, a in zip(self.levels, activities, strict=True)
        ]
        self._predicted_code = self.levels[0].render(self._predicted_activities[0])
        self._still_code = self.levels[0].render(activities[0])
        self._previous_code = code
        return StepResult(
            activities=activities,
            errors=errors,
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
