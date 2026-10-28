"""Motion cells (MT) that learn how V1's activity moves from one frame to the next.

Each motion cell type holds one non-negative spatial kernel, shared by all V1
channels and summing to one: where activity at a place goes on the next frame.
The type predicts V1's next activity as the current activity moved along its
kernel. At every place the types compete: a type's responsibility there is a
softmax, over types, of how well it predicted the transition just seen. Those
responsibilities are the motion cells' activities.

Learning is local and points at the future. Each type's kernel moves by the delta
rule, weighted by its responsibility: (V1's activity now − the type's prediction)
× V1's activity a frame ago, at the matching offset. That is competitive learning
on transitions. A newborn's kernels all say "nothing moves", with small random
differences, so which type comes to mean which motion is decided by experience.

What a type means is read from its own kernel: the displacement it predicts is
the kernel's centre of mass. No autograd is used.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F
from torch.nn.grad import conv2d_weight


class MotionLevel:
    def __init__(
        self,
        types: int = 16,
        reach: int = 3,
        temperature: float = 0.03,
        rate: float = 0.5,
        smoothing: int = 5,
        seed: int = 0,
        device="cpu",
    ):
        self.types, self.reach = types, reach
        self.temperature, self.rate, self.smoothing = temperature, rate, smoothing
        self.device = torch.device(device)
        side = 2 * reach + 1
        generator = torch.Generator().manual_seed(seed)
        # Newborn: almost all mass on "stays where it is", a little spread at random.
        kernels = 0.1 * torch.rand(types, side, side, generator=generator)
        kernels[:, reach, reach] += 0.9
        self.kernels = (kernels / kernels.flatten(1).sum(1)[:, None, None]).to(self.device)
        offsets = torch.arange(side, dtype=torch.float32) - reach
        self._dy, self._dx = torch.meshgrid(offsets, offsets, indexing="ij")
        self._dx, self._dy = self._dx.to(self.device), self._dy.to(self.device)

    # Prediction --------------------------------------------------------------

    def _weights(self) -> torch.Tensor:
        # conv2d correlates; flipping makes prediction(x) = Σ_d K(d) · activity(x − d).
        return self.kernels.flip(-1, -2)[:, None]

    def predictions(self, before: torch.Tensor) -> torch.Tensor:
        """``[B, M, C, H, W]``: each type's prediction of the next activity."""

        b, c, h, w = before.shape
        moved = F.conv2d(before.reshape(b * c, 1, h, w), self._weights(), padding=self.reach)
        return moved.reshape(b, c, self.types, h, w).transpose(1, 2)

    def responsibilities(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        """``[B, M, H, W]``: how much each type explains the transition at each place."""

        before, after = before.to(self.device), after.to(self.device)
        error = (after[:, None] - self.predictions(before)).square().sum(2)
        pad = self.smoothing // 2
        error = F.avg_pool2d(error, self.smoothing, 1, pad, count_include_pad=False)
        # Temperature relative to the typical error: the competition is scale-free.
        scale = self.temperature * error.mean().clamp_min(1e-12)
        return torch.softmax(-error / scale, dim=1)

    # Learning ----------------------------------------------------------------

    def learn(self, before: torch.Tensor, after: torch.Tensor, r: torch.Tensor) -> None:
        """Responsibility-weighted delta rule toward the transition seen."""

        before, after = before.to(self.device), after.to(self.device)
        b, c, h, w = before.shape
        residual = after[:, None] - self.predictions(before)  # [B, M, C, H, W]
        weighted = (r[:, :, None] * residual).transpose(1, 2).reshape(b * c, self.types, h, w)
        gradient = conv2d_weight(
            before.reshape(b * c, 1, h, w),
            (self.types, 1, 2 * self.reach + 1, 2 * self.reach + 1),
            weighted,
            padding=self.reach,
        )[:, 0].flip(-1, -2)
        # Normalised by the total presynaptic activity (as in V's rule), so that each
        # type's change stays in proportion to its share of the responsibility.
        energy = before.square().sum().clamp_min(1e-9)
        kernels = (self.kernels + self.rate * gradient / energy).clamp_min(0.0)
        self.kernels = kernels / kernels.flatten(1).sum(1).clamp_min(1e-12)[:, None, None]

    def step(self, before: torch.Tensor, after: torch.Tensor, learn: bool = True) -> torch.Tensor:
        with torch.no_grad():
            r = self.responsibilities(before, after)
            if learn:
                self.learn(before, after, r)
        return r

    # Reading the cells out -------------------------------------------------------

    def displacements(self) -> torch.Tensor:
        """``[M, 2]``: the (dx, dy) each type predicts, in grid positions."""

        dx = (self.kernels * self._dx).flatten(1).sum(1)
        dy = (self.kernels * self._dy).flatten(1).sum(1)
        return torch.stack([dx, dy], 1)

    def velocity(self, r: torch.Tensor) -> torch.Tensor:
        """``[B, 2, H, W]``: the population vector of the responsible types."""

        return torch.einsum("bmhw,mk->bkhw", r, self.displacements())

    def state_dict(self) -> dict:
        return {
            "kernels": self.kernels.cpu(),
            "settings": {
                "types": self.types,
                "reach": self.reach,
                "temperature": self.temperature,
                "rate": self.rate,
                "smoothing": self.smoothing,
            },
        }

    @classmethod
    def from_state(cls, state: dict, device="cpu") -> MotionLevel:
        level = cls(**state["settings"], device=device)
        level.kernels = state["kernels"].to(level.device)
        return level
