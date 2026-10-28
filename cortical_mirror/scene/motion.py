"""Predicting the next frame from how the scene's parts move.

Motion is sensed as dense optical flow between consecutive frames (Farnebäck):
an innate front end, in the role of the motion-sensitive area MT, not learned.
The scene is divided into parts. Each object node in view is a part, its region
its mask; whatever no node claims is divided into blocks of background. Each part
gets an affine map fitted to the flow inside it.

Raw motion does not simply continue from one frame to the next in real video,
but smoothed motion does (``reports/scene/dev-headroom.json``). So each part's
motion is averaged over its last few frame pairs and carried forward at half
strength.
"""

from __future__ import annotations

import cv2
import numpy as np

BLOCK = 32
HISTORY = 3
STRENGTH = 0.5
# A part needs this many pixels for its affine fit to mean anything.
SMALLEST = 20


def dense_flow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Where each pixel of a went in b, ``[H, W, 2]``."""

    u8 = [np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8) for x in (a, b)]
    return cv2.calcOpticalFlowFarneback(u8[0], u8[1], None, 0.5, 3, 15, 3, 5, 1.2, 0)


def warp(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """The image moved along a flow (sampled at p − flow(p))."""

    h, w = image.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    return cv2.remap(
        image,
        xx - flow[..., 0],
        yy - flow[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _fit(flow: np.ndarray, region: np.ndarray, origin) -> np.ndarray | None:
    """The affine map (3 × 2) from position relative to origin to flow, in a region."""

    ys, xs = np.nonzero(region)
    if len(xs) < SMALLEST:
        return None
    design = np.stack([xs - origin[0], ys - origin[1], np.ones(len(xs))], 1)
    return np.linalg.lstsq(design, flow[ys, xs], rcond=None)[0]


class PartMotion:
    def __init__(self, block: int = BLOCK, history: int = HISTORY, strength: float = STRENGTH):
        self.block, self.history, self.strength = block, history, strength
        self.maps: dict = {}
        self.regions: dict = {}

    @property
    def ready(self) -> bool:
        return bool(self.regions)

    def observe(self, previous: np.ndarray, now: np.ndarray, parts: dict) -> None:
        """Fit every part's motion from ``previous`` to ``now``. ``parts`` maps an
        object node's id to its region (a boolean mask) in ``now``."""

        flow = dense_flow(previous, now)
        h, w = now.shape
        claimed = np.zeros((h, w), bool)
        regions = {}
        for node, region in parts.items():
            if region.sum() >= SMALLEST:
                ys, xs = np.nonzero(region)
                regions[("node", node)] = (region, (float(xs.mean()), float(ys.mean())))
                claimed |= region
        for y in range(0, h, self.block):
            for x in range(0, w, self.block):
                region = np.zeros((h, w), bool)
                region[y : y + self.block, x : x + self.block] = True
                region &= ~claimed
                regions[("block", y, x)] = (region, (float(x), float(y)))
        maps = {}
        for key, (region, origin) in regions.items():
            fitted = _fit(flow, region, origin)
            if fitted is not None:
                # A node's motion is fitted relative to its own centre, so it carries
                # over as the node moves; a block's relative to its corner.
                maps[key] = [*self.maps.get(key, []), fitted][-self.history :]
        self.maps = maps
        self.regions = {k: v for k, v in regions.items() if k in maps}

    def field(self, shape) -> np.ndarray:
        """The motion expected next: each part's smoothed affine motion, damped."""

        out = np.zeros((*shape, 2), np.float32)
        for key, (region, origin) in self.regions.items():
            fitted = np.mean(self.maps[key], axis=0)
            ys, xs = np.nonzero(region)
            design = np.stack([xs - origin[0], ys - origin[1], np.ones(len(xs))], 1)
            out[ys, xs] = self.strength * (design @ fitted)
        return out

    def predict(self, now: np.ndarray) -> np.ndarray:
        """The next frame: the current one moved by the parts' expected motion."""

        if not self.ready:
            return now
        return warp(now, self.field(now.shape)).astype(np.float32)
