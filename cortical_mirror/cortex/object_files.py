"""Object files: each object's shape learned from what moves with it.

In an object's own frame, its pixels stay the same as it moves (its texture moves
with it), while the pixels around it show whatever background slides past. Each
place of the file keeps a running mean and variance of what it has seen (a local
rule that slowly forgets, so that early, poorly aligned views fade). The object's
shape is where the variance stays low, relative to the variance at the file's
border, which is background.

To keep its frame on the object, the file moves by the velocity it is told, then
corrects its position by its own prediction: within a few pixels, the place where
its learned appearance explains the image best (analysis by synthesis). That search
is engineered inference; the appearance and the shape are learned.

A file knows how sure it is: its fit is the error of its prediction, relative to the
variance it has learned to expect. A mature file that cannot explain what it sees
does not learn from it, and is not trusted until it finds its object again.
"""

from __future__ import annotations

import numpy as np


class ObjectFile:
    def __init__(
        self,
        x: float,
        y: float,
        radius: int = 24,
        search: int = 3,
        ratio: float = 0.25,
        mature: int = 30,
        align_after: int = 10,
        rate: float = 0.05,
        trust: float = 5.0,
    ):
        self.x, self.y, self.radius, self.search = float(x), float(y), radius, search
        self.ratio, self.mature, self.align_after = ratio, mature, align_after
        self.rate, self.trust = rate, trust
        self.fit = np.inf
        side = 2 * radius
        self.n = np.zeros((side, side), np.float32)
        self.mean = np.zeros((side, side), np.float32)
        self.var = np.zeros((side, side), np.float32)
        self.frames = 0

    # The file's frame on the image ------------------------------------------

    def _window(self, image: np.ndarray, cx: int, cy: int) -> np.ndarray:
        r = self.radius
        out = np.full((2 * r, 2 * r), np.nan, np.float32)
        h, w = image.shape
        x0, y0 = cx - r, cy - r
        ys = slice(max(y0, 0), min(y0 + 2 * r, h))
        xs = slice(max(x0, 0), min(x0 + 2 * r, w))
        if ys.start < ys.stop and xs.start < xs.stop:
            out[ys.start - y0 : ys.stop - y0, xs.start - x0 : xs.stop - x0] = image[ys, xs]
        return out

    def shape(self) -> np.ndarray:
        """Places of the file that belong to the object (in the file's frame)."""

        border = 3
        ring = np.ones_like(self.n, bool)
        ring[border:-border, border:-border] = False
        seen = self.n > 0
        if not (ring & seen).any():
            return np.zeros_like(seen)
        background = float(np.median(self.var[ring & seen]))
        return (self.var < self.ratio * background) & (self.n >= self.frames / 2) & seen

    @property
    def ready(self) -> bool:
        return self.frames >= self.mature

    @property
    def confident(self) -> bool:
        return self.ready and self.fit < self.trust

    # Following ------------------------------------------------------------------

    def _search(self, image: np.ndarray, x: float, y: float, search: int) -> None:
        """Put the file where, within ``search`` px of (x, y), its appearance
        explains the image best, and remember how well (its fit)."""

        weights = self.shape()
        if weights.sum() < 4:
            return
        cx, cy = int(round(x)), int(round(y))
        best, best_offset = np.inf, (0, 0)
        for dy in range(-search, search + 1):
            for dx in range(-search, search + 1):
                window = self._window(image, cx + dx, cy + dy)
                use = weights & ~np.isnan(window)
                if use.sum() < weights.sum() / 2:
                    continue
                cost = float(np.mean((window[use] - self.mean[use]) ** 2))
                if cost < best:
                    best, best_offset = cost, (dx, dy)
        if np.isfinite(best):
            self.x, self.y = cx + best_offset[0], cy + best_offset[1]
            self.fit = best / (float(np.mean(self.var[weights])) + 1e-4)

    def _align(self, image: np.ndarray) -> None:
        self._search(image, self.x, self.y, self.search)

    def relocate(self, image: np.ndarray, x: float, y: float, search: int = 8) -> None:
        """Look for the object again, near (x, y)."""

        self._search(image, x, y, search)

    def follow(self, image: np.ndarray, velocity=(0.0, 0.0)) -> None:
        """One frame: move by ``velocity`` (px per frame), correct by prediction, learn."""

        self.x += velocity[0]
        self.y += velocity[1]
        if self.frames >= self.align_after:
            self._align(image)
        if self.ready and self.fit >= self.trust:
            # What it cannot explain, it does not learn from.
            return
        window = self._window(image, int(round(self.x)), int(round(self.y)))
        seen = ~np.isnan(window)
        self.n[seen] += 1
        # The plain average at first, then a running one that forgets slowly.
        a = np.maximum(1.0 / self.n[seen], self.rate)
        delta = window[seen] - self.mean[seen]
        self.mean[seen] += a * delta
        self.var[seen] = (1 - a) * (self.var[seen] + a * delta**2)
        self.frames += 1
        if self.frames >= self.align_after:
            self._recentre()

    def _recentre(self) -> None:
        """Move the file's frame so that the shape it has learned sits in its middle."""

        shape = self.shape()
        if shape.sum() < 4:
            return
        ys, xs = np.nonzero(shape)
        dy = int(round(ys.mean() - (self.radius - 0.5)))
        dx = int(round(xs.mean() - (self.radius - 0.5)))
        if abs(dx) < 2 and abs(dy) < 2:
            return
        for name in ("n", "mean", "var"):
            a = getattr(self, name)
            moved = np.zeros_like(a)
            h, w = a.shape
            src_y, dst_y = slice(max(dy, 0), h + min(dy, 0)), slice(max(-dy, 0), h + min(-dy, 0))
            src_x, dst_x = slice(max(dx, 0), w + min(dx, 0)), slice(max(-dx, 0), w + min(-dx, 0))
            moved[dst_y, dst_x] = a[src_y, src_x]
            setattr(self, name, moved)
        self.x += dx
        self.y += dy

    def mask(self, height: int, width: int) -> np.ndarray:
        """The object's shape placed on the image."""

        out = np.zeros((height, width), bool)
        shape, r = self.shape(), self.radius
        x0, y0 = int(round(self.x)) - r, int(round(self.y)) - r
        ys = slice(max(y0, 0), min(y0 + 2 * r, height))
        xs = slice(max(x0, 0), min(x0 + 2 * r, width))
        if ys.start < ys.stop and xs.start < xs.stop:
            out[ys, xs] = shape[ys.start - y0 : ys.stop - y0, xs.start - x0 : xs.stop - x0]
        return out
