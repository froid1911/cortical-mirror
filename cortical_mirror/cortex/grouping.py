"""Grouping learned from common fate: an association field between V1 feature types.

``links[f, g, k]`` is how strongly V1 type f at one place belongs with type g at
offset k from it. A Hebbian rule, gated by common fate, learns it: pairs of
active places that move together (by the learned MT cells) strengthen the link
between their features, and pairs that move apart weaken it (Prodöhl, Würtz &
von der Malsburg, 2003). A newborn's field is zero.

The affinity between two places is then computed from their features alone, so
links learned from motion also group still frames. Groups are the connected
components of strong affinities.
"""

from __future__ import annotations

import numpy as np
import torch


def _shift(a: torch.Tensor, dy: int, dx: int, fill: float = 0.0) -> torch.Tensor:
    """``out[..., y, x] = a[..., y + dy, x + dx]``, with ``fill`` beyond the edge."""

    out = torch.full_like(a, fill)
    h, w = a.shape[-2:]
    ys, yd = slice(max(dy, 0), h + min(dy, 0)), slice(max(-dy, 0), h + min(-dy, 0))
    xs, xd = slice(max(dx, 0), w + min(dx, 0)), slice(max(-dx, 0), w + min(-dx, 0))
    out[..., yd, xd] = a[..., ys, xs]
    return out


class AssociationField:
    def __init__(
        self,
        types: int = 32,
        radius: int = 4,
        sigma: float = 1.0,
        rate: float = 0.1,
        moving: float = 0.5,
        device="cpu",
    ):
        self.types, self.radius, self.sigma = types, radius, sigma
        self.rate, self.moving = rate, moving
        side = 2 * radius + 1
        self.offsets = [
            (dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)
        ]
        self.links = torch.zeros(types, types, side * side, device=device)

    def learn(self, activity: torch.Tensor, velocity: torch.Tensor) -> None:
        """One frame: ``activity`` [F, H, W] (V1), ``velocity`` [2, H, W] (MT) on
        V1's grid. Only pairs of places that both move take part."""

        moving = velocity.norm(dim=0) > self.moving
        fates, pairs = [], []
        for dy, dx in self.offsets:
            if dy == 0 and dx == 0:
                fates.append(None)
                pairs.append(None)
                continue
            other_velocity = _shift(velocity, dy, dx)
            pair = moving & _shift(moving.float(), dy, dx).bool()
            fate = torch.exp(-(velocity - other_velocity).square().sum(0) / self.sigma**2)
            fates.append(fate)
            pairs.append(pair)
        counted = [p for p in pairs if p is not None]
        total = sum(int(p.sum()) for p in counted)
        if total == 0:
            return
        mean = sum(float(f[p].sum()) for f, p in zip(fates, pairs, strict=True) if p is not None)
        mean /= total
        for k, (dy, dx) in enumerate(self.offsets):
            if pairs[k] is None:
                continue
            weight = (fates[k] - mean) * pairs[k]
            other = _shift(activity, dy, dx)
            change = torch.einsum("fhw,ghw->fg", activity * weight, other)
            self.links[:, :, k] += self.rate * change / total * len(counted)

    def affinity(self, activity: torch.Tensor) -> torch.Tensor:
        """``[K, H, W]``: how strongly each place belongs with its neighbour at each
        offset, from features alone."""

        out = torch.zeros(len(self.offsets), *activity.shape[-2:], device=activity.device)
        for k, (dy, dx) in enumerate(self.offsets):
            if dy == 0 and dx == 0:
                continue
            other = _shift(activity, dy, dx)
            out[k] = torch.einsum("fhw,fg,ghw->hw", activity, self.links[:, :, k], other)
        return out

    def groups(
        self, activity: torch.Tensor, percentile: float = 80.0, minimum: int = 6
    ) -> torch.Tensor:
        """``[H, W]`` labels (0 = no group): connected components of affinities above
        the frame's percentile of positive affinities, at least ``minimum`` large."""

        h, w = activity.shape[-2:]
        active = activity.sum(0) > 1e-6
        affinity = self.affinity(activity)
        edges = []
        positive = []
        for k, (dy, dx) in enumerate(self.offsets):
            both = active & _shift(active.float(), dy, dx).bool()
            edges.append(both)
            positive.append(affinity[k][both & (affinity[k] > 0)])
        values = torch.cat(positive) if positive else torch.zeros(0)
        if values.numel() == 0:
            return torch.zeros(h, w, dtype=torch.long)
        threshold = torch.quantile(values.float().cpu(), percentile / 100.0).item()
        labels = torch.where(active, torch.arange(1, h * w + 1).reshape(h, w), 0).to(
            affinity.device
        )
        joined = [
            edges[k] & (affinity[k] >= threshold) & (affinity[k] > 0)
            for k in range(len(self.offsets))
        ]
        for _ in range(h + w):
            before = labels.clone()
            for k, (dy, dx) in enumerate(self.offsets):
                neighbour = _shift(labels, dy, dx)
                take = joined[k] & (neighbour > 0) & (neighbour < labels)
                labels = torch.where(take, neighbour, labels)
                # Links work both ways: the neighbour takes this place's label too.
                back = _shift((joined[k] & (labels > 0)).float(), -dy, -dx).bool()
                mine = _shift(labels, -dy, -dx)
                labels = torch.where(back & (mine > 0) & (mine < labels), mine, labels)
            if torch.equal(labels, before):
                break
        out = torch.zeros(h, w, dtype=torch.long)
        labels = labels.cpu()
        next_label = 1
        for value in torch.unique(labels):
            if value == 0:
                continue
            region = labels == value
            if int(region.sum()) >= minimum:
                out[region] = next_label
                next_label += 1
        return out


def adjusted_rand_index(truth: np.ndarray, found: np.ndarray) -> float:
    """The adjusted Rand index between two labellings of the same items."""

    truth, found = np.asarray(truth).ravel(), np.asarray(found).ravel()
    _, t = np.unique(truth, return_inverse=True)
    _, f = np.unique(found, return_inverse=True)
    table = np.zeros((t.max() + 1, f.max() + 1))
    np.add.at(table, (t, f), 1)

    def pairs(x):
        return float((x * (x - 1) / 2).sum())

    index = pairs(table)
    rows, columns = pairs(table.sum(1)), pairs(table.sum(0))
    total = pairs(np.array([truth.size]))
    expected = rows * columns / total if total else 0.0
    maximum = (rows + columns) / 2
    return (index - expected) / (maximum - expected) if maximum != expected else 1.0
