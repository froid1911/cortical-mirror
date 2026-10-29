"""A scene of learned objects that can be edited: what would the world look like if...

The objects are the motion slots (protocol 20), reading MT's fast path. The scene
also remembers the background: at every pixel the background slot owns, a running
average of what was seen there. That memory is what lets the scene say what lies
behind an object once it is deleted or moved (amodal completion). Where the
background was never seen, it can only guess, by inpainting, as a bitmap would.

With ``files``, each slot also keeps an object file (`object_files.py`) that learns
the object's shape from what moves with it; once the file is mature, its sharp mask
replaces the slot's coarse one.

Rendering by pasting and the inpainting fallback are engineered; the objects and
the memory come from what the cortex has seen.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from .object_files import ObjectFile


def inpaint(image: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """OpenCV's Telea inpainting, radius 5, on a [0, 1] float image."""

    if not hole.any():
        return image.copy()
    gray = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    filled = cv2.inpaint(gray, hole.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    out = image.copy()
    out[hole] = filled[hole].astype(np.float32) / 255.0
    return out


def shifted(a: np.ndarray, dx: int, dy: int, fill=0) -> np.ndarray:
    """``out[y, x] = a[y - dy, x - dx]``, with ``fill`` where nothing comes from."""

    out = np.full_like(a, fill)
    h, w = a.shape
    ys, yd = slice(max(-dy, 0), h - max(dy, 0)), slice(max(dy, 0), h - max(-dy, 0))
    xs, xd = slice(max(-dx, 0), w - max(dx, 0)), slice(max(dx, 0), w - max(-dx, 0))
    out[yd, xd] = a[ys, xs]
    return out


class LearnedScene:
    def __init__(
        self,
        cortex,
        slots,
        width: int = 256,
        height: int = 192,
        rate: float = 0.1,
        gap: int = 4,
        files: bool = False,
    ):
        self.cortex, self.slots = cortex, slots
        self.width, self.height, self.rate = width, height, rate
        self.memory = np.zeros((height, width), np.float32)
        self.seen = np.zeros((height, width), np.int32)
        self.past = deque(maxlen=gap + 1)
        self.frame: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.use_files = files
        self.files: dict[int, ObjectFile] = {}

    # Seeing ------------------------------------------------------------------

    def observe(self, image: np.ndarray) -> np.ndarray:
        """One frame ``[H, W]`` in [0, 1]; the slot labels on V1's grid out."""

        gray = F.avg_pool2d(torch.from_numpy(image)[None, None], 2).to(self.cortex.device)
        self.past.append(self.cortex.v1(gray))
        self.frame = image
        if len(self.past) < 2:
            self.labels = np.zeros(self.past[-1].shape[-2:], np.int64)
        else:
            velocity = self.cortex.between(self.past[0], self.past[-1])[0] / (len(self.past) - 1)
            self.labels = self.slots.step(velocity, self.past[-1][0]).numpy()
        if self.use_files and self.slots is not None:
            self._follow(image, self.labels)
        self.remember(image, self.labels)
        return self.labels

    def _follow(self, image: np.ndarray, labels: np.ndarray) -> None:
        """Each living slot's object file follows it; files of freed slots go."""

        alive = {k for k in range(1, len(self.slots.alive)) if bool(self.slots.alive[k])}
        for k in list(self.files):
            if k not in alive:
                del self.files[k]
        coarse = self._grid_masks(labels)
        # MT's velocity is in retinal px of the 128-wide view; the image is wider.
        scale = self.width / 128
        for k in alive:
            if k not in self.files:
                if k not in coarse:
                    continue
                ys, xs = np.nonzero(coarse[k])
                self.files[k] = ObjectFile(xs.mean(), ys.mean())
            u = self.slots.u[k].tolist()
            self.files[k].follow(image, velocity=(u[0] * scale, u[1] * scale))

    def masks(self, labels: np.ndarray) -> dict[int, np.ndarray]:
        """Each object slot's region on the pixels: its object file's shape once
        the file is mature, else the slot's region on V1's grid."""

        masks = self._grid_masks(labels)
        for k, file in self.files.items():
            if k in masks and file.ready:
                sharp = file.mask(self.height, self.width)
                if sharp.any():
                    masks[k] = sharp
        return masks

    def _grid_masks(self, labels: np.ndarray) -> dict[int, np.ndarray]:
        block = self.width // labels.shape[1]
        pixels = np.kron(labels, np.ones((block, block), np.int64))[: self.height, : self.width]
        return {int(k): pixels == k for k in np.unique(pixels) if k > 0}

    def remember(self, image: np.ndarray, labels: np.ndarray) -> None:
        """Update the background memory where the background slot owns the place."""

        owned = np.ones((self.height, self.width), bool)
        for mask in self.masks(labels).values():
            owned &= ~mask
        first = owned & (self.seen == 0)
        later = owned & (self.seen > 0)
        self.memory[first] = image[first]
        self.memory[later] += self.rate * (image[later] - self.memory[later])
        self.seen[owned] += 1

    # Explaining by changing -------------------------------------------------

    def edit(self, frame: np.ndarray, labels: np.ndarray, delete=frozenset(), move=None):
        """The frame as it would look with slots deleted or moved by (dx, dy) px."""

        move = move or {}
        masks = self.masks(labels)
        out = frame.copy()
        removed = np.zeros_like(out, bool)
        for k in set(delete) | set(move):
            removed |= masks.get(k, np.zeros_like(removed))
        known = removed & (self.seen > 0)
        out[known] = self.memory[known]
        out = inpaint(out, removed & ~known)
        for k, (dx, dy) in move.items():
            if k not in masks:
                continue
            target = shifted(masks[k], dx, dy, False)
            out[target] = shifted(frame, dx, dy)[target]
        return out
