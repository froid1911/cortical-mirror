"""The eye as a node of the scene graph: what its own movements do to the view.

The graph keeps the scene on a canvas larger than any view, in its own world
coordinates. The eye is a node with a gaze on that canvas. Before each frame the
graph is told the motor command just issued (the efference copy). Its learned
motor map says how far that command moves the gaze; the graph moves its gaze by
that much. It then registers the new view against the canvas. Whatever shift
remains is not the eye's doing, so it is the world moving: the canvas and every
object node move by it.

The motor map learns by the delta rule on that remainder, one entry per command.
A newborn's map is all zeros. It therefore takes every shift of its view for the
world moving, including the shifts its own saccades cause.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..cortex.eye import COMMANDS
from .graph import SceneGraph, _as_gray

SACCADE = 32
CANVAS = (768, 576)
# Largest shift between prediction and view that registration looks for.
REACH = 48
COARSE = 4


def look(image: np.ndarray, gaze, view: tuple[int, int]) -> np.ndarray:
    """The view ``[h, w]`` of a world image centred on the gaze."""

    x, y = int(round(gaze[0])), int(round(gaze[1]))
    w, h = view
    return image[y - h // 2 : y - h // 2 + h, x - w // 2 : x - w // 2 + w]


def _shift(a: np.ndarray, dx: int, dy: int, fill: float = 0.0) -> np.ndarray:
    """Content moved by (dx, dy), ``fill`` coming in."""

    out = np.full_like(a, fill)
    h, w = a.shape
    out[max(dy, 0) : h + min(dy, 0), max(dx, 0) : w + min(dx, 0)] = a[
        max(-dy, 0) : h + min(-dy, 0), max(-dx, 0) : w + min(-dx, 0)
    ]
    return out


class EyeGraph:
    def __init__(
        self,
        view: tuple[int, int] = (256, 192),
        persistence: float = 30.0,
        learn_persistence: bool = False,
        learn_motor: bool = True,
        motor: np.ndarray | None = None,
        fovea: int | None = None,
    ):
        self.view = view
        # Side of the sharp central square; the rest of the view is seen blurred.
        self.fovea = fovea
        self.fovea_size = fovea or min(view)
        self.inner = SceneGraph(*view, persistence=persistence, learn_persistence=learn_persistence)
        self.learn_motor = learn_motor
        self.motor = np.zeros((len(COMMANDS), 2)) if motor is None else np.array(motor, float)
        self.start = (CANVAS[0] // 2, CANVAS[1] // 2)
        self.reset_scene()

    def reset_scene(self) -> None:
        """A new scene: nothing known about it; the motor map is kept."""

        self.inner.reset()
        self.background = np.zeros(CANVAS[::-1], np.float32)
        self.confidence = np.zeros(CANVAS[::-1], np.float32)
        self.front = np.zeros(CANVAS[::-1], np.float32)
        self.gaze = self.start
        self.command = 0
        self.world_shift = np.zeros(2)
        # How badly the graph has been predicting each place, and the error in the
        # fovea frame by frame (what attention learns from).
        self.error = np.zeros(CANVAS[::-1], np.float32)
        self.fovea_errors: list[float] = []

    # Where the view lies on the canvas.

    def _origin(self, gaze=None) -> tuple[int, int]:
        x, y = self.gaze if gaze is None else gaze
        return int(round(x)) - self.view[0] // 2, int(round(y)) - self.view[1] // 2

    def _window(self, a: np.ndarray, origin) -> np.ndarray:
        x, y = origin
        return a[y : y + self.view[1], x : x + self.view[0]]

    def _register(self, image: np.ndarray, origin) -> tuple[int, int] | None:
        """The shift (dx, dy) at which the canvas around ``origin`` best matches the
        view, weighted by how sure the canvas is; None if that is not clear: too
        little is known, or the best match is at the edge of the search, or it is
        not much better than a typical shift."""

        x, y = origin
        w, h = self.view
        if not (
            REACH <= x and REACH <= y and x + w + REACH <= CANVAS[0] and y + h + REACH <= CANVAS[1]
        ):
            return None
        region = (slice(y - REACH, y + h + REACH), slice(x - REACH, x + w + REACH))
        known, sure = self.background[region], self.confidence[region]
        if sure.mean() < 0.1:
            return None

        def search(view, known, sure, reach, step):
            windows = sliding_window_view(known, view.shape)[::step, ::step]
            weights = sliding_window_view(sure, view.shape)[::step, ::step]
            mass = weights.sum((-1, -2))
            cost = (weights * (windows - view) ** 2).sum((-1, -2)) / np.maximum(mass, 1e-6)
            cost[mass < 0.25 * view.size] = np.inf
            best = np.unravel_index(int(np.argmin(cost)), cost.shape)
            finite = cost[np.isfinite(cost)]
            typical = float(np.median(finite)) if finite.size else np.inf
            return best[1] * step - reach, best[0] * step - reach, float(cost[best]), typical

        small = (slice(None, None, COARSE), slice(None, None, COARSE))
        coarse_x, coarse_y, cost, typical = search(
            image[small], known[small], sure[small], REACH // COARSE, 1
        )
        edge = REACH // COARSE
        if not np.isfinite(cost) or cost > 0.5 * typical or edge in (abs(coarse_x), abs(coarse_y)):
            return None
        coarse_x, coarse_y = coarse_x * COARSE, coarse_y * COARSE
        fine = 3
        fx, fy = REACH + coarse_x - fine, REACH + coarse_y - fine
        if (
            fx < 0
            or fy < 0
            or fx + 2 * fine + w > known.shape[1]
            or fy + 2 * fine + h > known.shape[0]
        ):
            return coarse_x, coarse_y
        part = (slice(fy, fy + h + 2 * fine), slice(fx, fx + w + 2 * fine))
        dx, dy, _, _ = search(image, known[part], sure[part], fine, 1)
        return coarse_x + dx, coarse_y + dy

    def foveate(self, image: np.ndarray) -> np.ndarray:
        """Sharp in the central square, blurred outside, with a soft border."""

        if self.fovea is None:
            return image
        h, w = image.shape
        weight = np.zeros_like(image)
        half = self.fovea // 2
        weight[h // 2 - half : h // 2 + half, w // 2 - half : w // 2 + half] = 1.0
        weight = cv2.GaussianBlur(weight, (0, 0), 4.0)
        weight[h // 2 - half + 8 : h // 2 + half - 8, w // 2 - half + 8 : w // 2 + half - 8] = 1.0
        return weight * image + (1 - weight) * cv2.GaussianBlur(image, (0, 0), 3.0)

    def observe(self, image: np.ndarray, command: int = 0) -> dict:
        """One view, and the command issued just before it (the efference copy)."""

        image = self.foveate(_as_gray(image, *self.view))
        self.command = command
        before = self._origin()
        planned = self.motor[command]
        self.gaze = (self.gaze[0] + planned[0], self.gaze[1] + planned[1])
        origin = self._origin()
        remainder = (0, 0)
        if self.inner.background is not None:
            remainder = self._register(image, origin)
            if remainder is None:
                # Not known well enough to tell: nothing is attributed, nothing learned.
                remainder = (0, 0)
            elif self.learn_motor:
                self.motor[command] += 0.5 * np.asarray(remainder, float)
            # The remainder is the world moving: the view's content now sits where
            # the canvas had it at origin + remainder, so the canvas moves back by it.
            dx, dy = remainder
            self.background = _shift(self.background, -dx, -dy)
            self.confidence = _shift(self.confidence, -dx, -dy)
            self.front = _shift(self.front, -dx, -dy)
            self.world_shift -= np.asarray(remainder, float)
            # In the view's frame everything moved by the eye's movement plus the world's.
            vx, vy = origin[0] - before[0] + dx, origin[1] - before[1] + dy
            self._move_inner(-vx, -vy, image)
        self.inner.background = (
            None if self.inner.background is None else self._window(self.background, origin).copy()
        )
        if self.inner.background is not None:
            self.inner.confidence = self._window(self.confidence, origin).copy()
            self.inner.front = self._window(self.front, origin).copy()
        if self.inner.background is not None:
            # What the graph expected here, scored before it sees the frame.
            miss = (self.inner.expect() - image) ** 2
            self._window(self.error, origin)[:] = (
                0.7 * self._window(self.error, origin) + 0.3 * miss
            )
            w, h = self.view
            half = self.fovea_size // 2
            centre = (slice(h // 2 - half, h // 2 + half), slice(w // 2 - half, w // 2 + half))
            self.fovea_errors.append(float(miss[centre].mean()))
        result = self.inner.observe(image)
        cx, cy = self.view[0] / 2, self.view[1] / 2
        for node in self.inner.nodes:
            if abs(node.x - cx) < self.fovea_size / 2 and abs(node.y - cy) < self.fovea_size / 2:
                node.looked += 1
        self._window(self.background, origin)[:] = self.inner.background
        self._window(self.confidence, origin)[:] = self.inner.confidence
        self._window(self.front, origin)[:] = self.inner.front
        return {**result, "world_moved": (-remainder[0], -remainder[1])}

    def _move_inner(self, dx: int, dy: int, image: np.ndarray) -> None:
        """Shift the inner graph's view-frame state by (dx, dy)."""

        if dx == 0 and dy == 0:
            return
        for node in self.inner.nodes:
            node.x, node.y = node.x + dx, node.y + dy
            node.trail = [(x + dx, y + dy) for x, y in node.trail or []]
            if node.origin:
                node.origin = (node.origin[0] + dx, node.origin[1] + dy)
        for node, _ in self.inner.memory:
            node.x, node.y = node.x + dx, node.y + dy
        # Past views, moved the same way; where nothing was seen, the present stands in.
        self.inner._recent = [
            np.where(_shift(np.ones_like(f), dx, dy) > 0, _shift(f, dx, dy), image)
            for f in self.inner._recent
        ]
        if self.inner.previous is not None:
            seen = _shift(np.ones_like(self.inner.previous), dx, dy) > 0
            self.inner.previous = np.where(seen, _shift(self.inner.previous, dx, dy), image)

    def predict_view(self, command: int) -> tuple[np.ndarray, np.ndarray]:
        """What the graph expects to see after ``command``, from its canvas alone, and
        where it is sure of that."""

        planned = self.motor[command]
        origin = self._origin((self.gaze[0] + planned[0], self.gaze[1] + planned[1]))
        return (
            self._window(self.background, origin).copy(),
            self._window(self.confidence, origin) > 0.5,
        )

    def world_positions(self) -> list[tuple[int, float, float]]:
        """Every object node's position on the canvas: where it is in the world."""

        x0, y0 = self._origin()
        return [(n.id, n.x + x0, n.y + y0) for n in self.inner.nodes]

    def render(self) -> np.ndarray:
        return self.inner.render()

    def to_json(self) -> dict:
        data = self.inner.to_json()
        x0, y0 = self._origin()
        for node in data["nodes"]:
            node["world_x"], node["world_y"] = round(node["x"] + x0, 1), round(node["y"] + y0, 1)
        data["self"] = {
            "kind": "eye",
            "gaze": [round(self.gaze[0], 1), round(self.gaze[1], 1)],
            "command": int(self.command),
            "motor": [[round(float(v), 1) for v in row] for row in self.motor],
            "world_shift": [round(float(v), 1) for v in self.world_shift],
        }
        return data
