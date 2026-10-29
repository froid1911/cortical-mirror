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

    def velocity(self, r: torch.Tensor, sharpness: float = 1.0) -> torch.Tensor:
        """``[B, 2, H, W]``: the population vector of the responsible types. With
        ``sharpness`` > 1 the read-out leans on the types that clearly own a place
        (responsibilities raised to that power and renormalised)."""

        if sharpness != 1.0:
            r = r.pow(sharpness)
            r = r / r.sum(1, keepdim=True).clamp_min(1e-12)
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


class MotionCortex:
    """V1 (frozen, phase 1's adult or a newborn) and MT at two scales.

    The fine level works on V1's grid; the coarse one on V1 pooled to half
    resolution, so that fast motion and camera pans are within reach. Where the
    coarse level sees motion beyond the fine level's reach, its velocity is used.
    """

    def __init__(
        self,
        v1: str | None = None,
        types: int = 16,
        reach: int = 3,
        seed: int = 0,
        device="cpu",
        settle: int = 20,
    ):
        from .engine import RETINA_SIZE, CortexEngine
        from .retina import Retina

        self.device = torch.device(device)
        engine = (
            CortexEngine.load(v1, device=device, learn=False)
            if v1
            else CortexEngine(device=device, learn=False)
        )
        self.v1_path = str(v1) if v1 else None
        self.hierarchy = engine.hierarchy
        self.retina = Retina(*RETINA_SIZE, device=self.device)
        self.settle = settle
        self.fine = MotionLevel(types=types, reach=reach, seed=seed, device=device)
        self.coarse = MotionLevel(types=types, reach=reach, seed=seed + 1, device=device)
        self.silenced = False
        self.previous: torch.Tensor | None = None
        self.stride = engine.hierarchy.configs[0].stride

    def v1(self, gray: torch.Tensor) -> torch.Tensor:
        """V1's activity for frames ``[B, 1, 96, 128]`` (settled from rest, frozen)."""

        return self.hierarchy.settle(self.retina.encode(gray.to(self.device)), self.settle)[0]

    def reset(self) -> None:
        self.previous = None

    @torch.no_grad()
    def step(self, gray: torch.Tensor, learn: bool = False) -> torch.Tensor | None:
        """One frame in; the velocity of what moved since the last one out, in retinal
        pixels per frame, ``[B, 2, H, W]`` at the retina's size (None on the first)."""

        now = self.v1(gray)
        before, self.previous = self.previous, now
        if before is None or before.shape != now.shape:
            return None
        # The transition just seen, kept for measurements of the cells themselves.
        self._last_pair = (before, now)
        height, width = gray.shape[-2:]
        if self.silenced:
            return torch.zeros(now.shape[0], 2, height, width, device=self.device)
        # The fine types' responsibilities, kept for the efference copy to learn from.
        self.last_fine = self.fine.step(before, now, learn)
        fine = self.fine.velocity(self.last_fine) * self.stride
        pooled_before, pooled_now = F.avg_pool2d(before, 2), F.avg_pool2d(now, 2)
        coarse = self.coarse.velocity(self.coarse.step(pooled_before, pooled_now, learn))
        coarse = F.interpolate(coarse * 2 * self.stride, size=fine.shape[-2:], mode="bilinear")
        reach = self.fine.reach * self.stride
        velocity = torch.where(coarse.norm(dim=1, keepdim=True) > reach, coarse, fine)
        return F.interpolate(velocity, size=(height, width), mode="bilinear")

    def save(self, path) -> None:
        torch.save(
            {
                "v1": self.v1_path,
                "v1_state": self.hierarchy.state_dict(),
                "fine": self.fine.state_dict(),
                "coarse": self.coarse.state_dict(),
                "settle": self.settle,
            },
            path,
        )

    @classmethod
    def load(cls, path, device="cpu") -> MotionCortex:
        state = torch.load(path, map_location="cpu", weights_only=True)
        cortex = cls(
            types=state["fine"]["settings"]["types"],
            reach=state["fine"]["settings"]["reach"],
            device=device,
            settle=state["settle"],
        )
        cortex.v1_path = state["v1"]
        cortex.hierarchy.load_state_dict(state["v1_state"])
        cortex.fine = MotionLevel.from_state(state["fine"], device)
        cortex.coarse = MotionLevel.from_state(state["coarse"], device)
        return cortex
