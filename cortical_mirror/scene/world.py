"""A synthetic world of textured objects, with the truth kept for scoring only.

Objects of random smooth shape and their own 1/f texture drift across a 1/f
background and bounce off the world's edges. Static bars may stand in front of
everything. The learner sees only rendered frames; ``truth()`` exists so that
protocols can score what it inferred, and ``render(moved=..., deleted=...)``
shows what the world would look like if it were changed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

SPRITE = 48


def texture(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """Natural-statistics (1/f) texture in [0, 1]."""

    image = np.zeros((height, width), np.float32)
    for scale in (1, 2, 4, 8, 16):
        size = (max(2, width // scale), max(2, height // scale))
        noise = rng.random(size[::-1]).astype(np.float32)
        image += scale * cv2.resize(noise, (width, height), interpolation=cv2.INTER_LINEAR)
    image -= image.min()
    return image / max(float(image.max()), 1e-6)


def blob(rng: np.random.Generator, size: int = SPRITE) -> np.ndarray:
    """A random smooth shape, ``[size, size]`` bool, centred."""

    radius = rng.uniform(9, 18)
    harmonics = [(k, rng.uniform(0, 0.25), rng.uniform(0, 2 * math.pi)) for k in (2, 3, 4)]
    yy, xx = np.mgrid[:size, :size] - (size - 1) / 2
    angle = np.arctan2(yy, xx)
    edge = radius * (1 + sum(a * np.cos(k * angle + p) for k, a, p in harmonics))
    return np.hypot(xx, yy) <= edge


@dataclass
class Thing:
    id: int
    texture: np.ndarray
    mask: np.ndarray
    x: float
    y: float
    vx: float
    vy: float
    depth: int


class SpriteWorld:
    def __init__(
        self,
        seed: int = 0,
        width: int = 256,
        height: int = 192,
        objects: int = 3,
        occluders: int = 0,
        speed: tuple[float, float] = (0.5, 2.5),
    ):
        self.width, self.height = width, height
        rng = np.random.default_rng(seed)
        self.background = 0.15 + 0.5 * texture(height, width, rng)
        self.objects: list[Thing] = []
        for index in range(objects):
            # Each object has its own brightness, so objects differ from the background.
            level = rng.uniform(0.35, 0.65)
            surface = level + 0.35 * (texture(SPRITE, SPRITE, rng) - 0.5)
            angle, pace = rng.uniform(0, 2 * math.pi), rng.uniform(*speed)
            self.objects.append(
                Thing(
                    id=index,
                    texture=surface.clip(0, 1).astype(np.float32),
                    mask=blob(rng),
                    x=float(rng.uniform(SPRITE / 2, width - SPRITE / 2)),
                    y=float(rng.uniform(SPRITE / 2, height - SPRITE / 2)),
                    vx=pace * math.cos(angle),
                    vy=pace * math.sin(angle),
                    depth=index,
                )
            )
        self.occluders: list[Thing] = []
        for index in range(occluders):
            bar = np.zeros((SPRITE, SPRITE), bool)
            bar[:, SPRITE // 2 - 7 : SPRITE // 2 + 7] = True
            self.occluders.append(
                Thing(
                    id=objects + index,
                    texture=np.full((SPRITE, SPRITE), 0.9, np.float32),
                    mask=bar,
                    x=width * (index + 1) / (occluders + 1),
                    y=height / 2,
                    vx=0.0,
                    vy=0.0,
                    depth=objects + index,
                )
            )

    def step(self) -> None:
        # Objects bounce before their centre reaches the edge, so most of them stays in view.
        margin = SPRITE // 2 - 4
        for thing in self.objects:
            thing.x += thing.vx
            thing.y += thing.vy
            if not margin <= thing.x <= self.width - margin:
                thing.vx = -thing.vx
                thing.x = min(max(thing.x, margin), self.width - margin)
            if not margin <= thing.y <= self.height - margin:
                thing.vy = -thing.vy
                thing.y = min(max(thing.y, margin), self.height - margin)

    def _stamp(self, thing: Thing, offset=(0, 0)) -> tuple[np.ndarray, np.ndarray]:
        """The thing's texture and mask placed in world coordinates."""

        texture_out = np.zeros((self.height, self.width), np.float32)
        mask_out = np.zeros((self.height, self.width), bool)
        left = int(round(thing.x + offset[0])) - SPRITE // 2
        top = int(round(thing.y + offset[1])) - SPRITE // 2
        x0, y0 = max(left, 0), max(top, 0)
        x1, y1 = min(left + SPRITE, self.width), min(top + SPRITE, self.height)
        if x0 < x1 and y0 < y1:
            sx, sy = x0 - left, y0 - top
            texture_out[y0:y1, x0:x1] = thing.texture[sy : sy + y1 - y0, sx : sx + x1 - x0]
            mask_out[y0:y1, x0:x1] = thing.mask[sy : sy + y1 - y0, sx : sx + x1 - x0]
        return texture_out, mask_out

    def _layers(self, moved=None, deleted=None) -> list[tuple[Thing, np.ndarray, np.ndarray]]:
        moved, deleted = moved or {}, deleted or set()
        things = [t for t in self.objects + self.occluders if t.id not in deleted]
        return [
            (t, *self._stamp(t, moved.get(t.id, (0, 0))))
            for t in sorted(things, key=lambda t: t.depth)
        ]

    def background_view(self) -> np.ndarray:
        return self.background.copy()

    def render(self, moved: dict | None = None, deleted: set | None = None) -> np.ndarray:
        """The world, ``[H, W]`` float32 in [0, 1]; edits apply to this picture only."""

        image = self.background.copy()
        for _, surface, mask in self._layers(moved, deleted):
            image[mask] = surface[mask]
        return image

    def frame(self) -> np.ndarray:
        gray = np.rint(self.render() * 255).astype(np.uint8)
        return np.repeat(gray[..., None], 3, axis=2)

    def truth(self) -> list[dict]:
        """Per object: id, full mask, visible mask (after nearer things), position, depth."""

        layers = self._layers()
        out = []
        for index, (thing, _, mask) in enumerate(layers):
            if thing not in self.objects:
                continue
            visible = mask.copy()
            for _, _, nearer in layers[index + 1 :]:
                visible &= ~nearer
            out.append(
                {
                    "id": thing.id,
                    "full": mask,
                    "visible": visible,
                    "x": thing.x,
                    "y": thing.y,
                    "depth": thing.depth,
                }
            )
        return out
