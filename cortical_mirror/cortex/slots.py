"""Objects as slots found by learned motion, held by learned links.

A background slot and a few object slots share V1's grid. Each holds a soft mask
(its responsibility at every place) and a velocity. Every frame, a few rounds of
EM assign places to slots:

- prior: the slot's last mask, moved by its velocity and blurred a little (memory
  diffuses unless evidence keeps it sharp);
- motion evidence: how well the slot's velocity matches MT's at the place;
- link evidence: the learned affinity (protocol 19's association field) between
  the place's features and those of the places the slot holds.

Moving places the background explains worst start new slots; slots that shrink to
nothing are freed; touching slots that move alike merge. The EM machinery, birth,
death and merging are engineered inference; what it reads (V1, MT, the links) is
learned.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
from torch.nn import functional as F

from .grouping import AssociationField, _shift


def components(mask: np.ndarray) -> np.ndarray:
    """4-connected components of ``mask``, labelled 1, 2, ... (0 elsewhere)."""

    labels = np.zeros(mask.shape, np.int64)
    h, w = mask.shape
    current = 0
    for y, x in zip(*np.nonzero(mask), strict=True):
        if labels[y, x]:
            continue
        current += 1
        labels[y, x] = current
        queue = deque([(y, x)])
        while queue:
            cy, cx = queue.popleft()
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = current
                    queue.append((ny, nx))
    return labels


class MotionSlots:
    def __init__(
        self,
        slots: int = 6,
        stride: int = 2,
        sigma: float = 0.5,
        beta: float = 2.0,
        blur: float = 0.2,
        rounds: int = 3,
        moving: float = 0.2,
        birth: int = 6,
        death: float = 3.0,
        patience: int = 5,
        field: AssociationField | None = None,
        device="cpu",
    ):
        self.slots, self.stride, self.sigma, self.beta = slots, stride, sigma, beta
        self.blur, self.rounds, self.moving = blur, rounds, moving
        self.birth, self.death, self.patience = birth, death, patience
        self.field, self.device = field, torch.device(device)
        self.reset()

    def reset(self) -> None:
        self.q: torch.Tensor | None = None
        self.u = torch.zeros(self.slots + 1, 2, device=self.device)
        self.alive = torch.zeros(self.slots + 1, dtype=torch.bool, device=self.device)
        self.alive[0] = True
        self.low = [0] * (self.slots + 1)

    # Pieces ------------------------------------------------------------------

    def _moved(self, q: torch.Tensor) -> torch.Tensor:
        """Each mask moved by its slot's velocity (in V1 positions), then blurred."""

        s, h, w = q.shape
        shift = self.u / self.stride
        ys = torch.linspace(-1, 1, h, device=q.device)
        xs = torch.linspace(-1, 1, w, device=q.device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        # out(x) = q(x − d): sample each mask d positions back.
        sx = gx[None] - shift[:, 0, None, None] * 2 / max(w - 1, 1)
        sy = gy[None] - shift[:, 1, None, None] * 2 / max(h - 1, 1)
        grid = torch.stack([sx, sy], -1)
        moved = F.grid_sample(q[:, None], grid, align_corners=True, padding_mode="border")
        side = torch.tensor([self.blur / 2, 1 - self.blur, self.blur / 2], device=q.device)
        kernel = torch.outer(side, side)[None, None]
        return F.conv2d(F.pad(moved, (1, 1, 1, 1), mode="replicate"), kernel)[:, 0]

    def _links(self, activity: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """``[S, H, W]``: the mean normalised affinity between each place and the
        places each slot holds."""

        affinity = self.field.affinity(activity)
        affinity = affinity / affinity.abs().mean().clamp_min(1e-12)
        out = torch.zeros_like(q)
        for k, (dy, dx) in enumerate(self.field.offsets):
            if dy == 0 and dx == 0:
                continue
            out += affinity[k][None] * _shift(q, dy, dx)
        return out / (len(self.field.offsets) - 1)

    def _assign(self, prior, velocity, activity) -> torch.Tensor:
        for _ in range(self.rounds):
            error = (velocity[None] - self.u[:, :, None, None]).square().sum(1)
            evidence = prior * torch.exp(-error / (2 * self.sigma**2))
            if self.field is not None and self.beta > 0 and activity is not None:
                links = self._links(activity, self.q)
                evidence = evidence * torch.exp(self.beta * (links - links.amax(0)))
            evidence = evidence * self.alive[:, None, None]
            self.q = evidence / evidence.sum(0, keepdim=True).clamp_min(1e-12)
            mass = self.q.sum((1, 2))
            u = torch.einsum("shw,khw->sk", self.q, velocity) / mass[:, None].clamp_min(1e-9)
            self.u = torch.where(self.alive[:, None], u, torch.zeros_like(u))
        return self.q

    def _free(self, k: int) -> None:
        self.q[0] += self.q[k]
        self.q[k] = 0
        self.u[k] = 0
        self.alive[k] = False
        self.low[k] = 0

    def _merge(self, labels: torch.Tensor) -> None:
        held = [k for k in range(1, self.slots + 1) if self.alive[k]]
        for i in held:
            for j in held:
                if j <= i or not (self.alive[i] and self.alive[j]):
                    continue
                if float((self.u[i] - self.u[j]).norm()) > self.sigma:
                    continue
                grown = F.max_pool2d((labels == i).float()[None, None], 3, 1, 1)[0, 0] > 0
                if bool((grown & (labels == j)).any()):
                    self.q[i] += self.q[j]
                    self.q[j] = 0
                    self._free(j)

    def _births(self, velocity: torch.Tensor, labels: torch.Tensor) -> None:
        candidates = (velocity.norm(dim=0) > self.moving) & (labels == 0)
        found = components(candidates.cpu().numpy())
        sizes = np.bincount(found.ravel())[1:]
        for index in np.argsort(-sizes):
            free = [k for k in range(1, self.slots + 1) if not self.alive[k]]
            if sizes[index] < self.birth or not free:
                break
            k = free[0]
            region = torch.from_numpy(found == index + 1).to(self.q.device)
            self.q[:, region] = 0
            self.q[k, region] = 1
            self.u[k] = velocity[:, region].mean(1)
            self.alive[k] = True
            self.low[k] = 0

    # The step ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, velocity: torch.Tensor, activity: torch.Tensor | None = None) -> torch.Tensor:
        """One frame: MT's ``velocity`` [2, H, W] (retinal px per frame) and V1's
        ``activity`` [F, H, W] on V1's grid in; labels [H, W] out (0 = background)."""

        velocity = velocity.to(self.device)
        activity = activity.to(self.device) if activity is not None else None
        if self.q is None:
            self.q = torch.zeros(self.slots + 1, *velocity.shape[-2:], device=self.device)
            self.q[0] = 1
        prior = self._moved(self.q) + 1e-3
        self._assign(prior, velocity, activity)
        labels = self.q.argmax(0)
        self._merge(labels)
        for k in range(1, self.slots + 1):
            if not self.alive[k]:
                continue
            self.low[k] = self.low[k] + 1 if float(self.q[k].sum()) < self.death else 0
            if self.low[k] >= self.patience:
                self._free(k)
        labels = self.q.argmax(0)
        self._births(velocity, labels)
        return self.q.argmax(0).cpu()
