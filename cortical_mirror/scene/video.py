"""The scene graph on video filmed by a moving camera.

There is no efference copy here: every global shift of the view is the camera's.
The graph registers each frame against its canvas (the eye graph with no
commands), so the background stays put and only objects move. Its expectation
of the next frame assumes the camera keeps moving as it just did, and the graph
predicts only what it explains: its objects, and the background where it is
sure. Everywhere else it predicts the last frame, moved with the camera.
"""

from __future__ import annotations

import cv2
import numpy as np

from .eye import EyeGraph, _shift, recentre
from .front import LuminanceFront
from .graph import _as_gray


class VideoGraph:
    def __init__(self, size: tuple[int, int] = (256, 192), front=None, persistence: float = 30.0):
        self.size = size
        self.front = front or LuminanceFront()
        self.eye = EyeGraph(size, persistence=persistence, learn_motor=False)
        # The camera's motion, as the shift of the view's content per frame, and how
        # far the camera has moved in all.
        self.velocity = np.zeros(2)
        self.travelled = np.zeros(2)
        self.previous: np.ndarray | None = None
        self._previous_code: np.ndarray | None = None

    @property
    def graph(self):
        return self.eye.inner

    def observe(self, frame: np.ndarray) -> None:
        gray = _as_gray(frame, *self.size)
        code = self.front.code(gray)
        # The camera's own motion since the last frame, from the frames themselves:
        # handed to the eye graph as what its "eye" did, so that registration against
        # the canvas only has to find what is left.
        shift = np.zeros(2)
        if self._previous_code is not None:
            (dx, dy), response = cv2.phaseCorrelate(
                self._previous_code.astype(np.float64), code.astype(np.float64)
            )
            if response > 0.05 and abs(dx) < 48 and abs(dy) < 48:
                shift = np.array([dx, dy])
        self.eye.motor[0] = -np.round(shift)
        gaze, before = self.eye.gaze, self.eye.world_shift.copy()
        self.eye.observe(code, 0, shade=gray)
        # In the view, content moved by minus the eye's movement plus the world's.
        content = -(np.array(self.eye.gaze) - np.array(gaze)) + (self.eye.world_shift - before)
        self.velocity += 0.5 * (content - self.velocity)
        self.travelled -= content
        recentre(self.eye)
        self.previous, self._previous_code = gray, code

    def expect(self) -> np.ndarray:
        """The next frame, ``[H, W]`` brightness."""

        if self.previous is None:
            return np.zeros(self.size[::-1], np.float32)
        graph = self.graph
        explained = graph.confidence > 0.5
        for node in graph.nodes:
            if node.state == "seen":
                explained |= graph._alpha(node) > 0.5
        # Only where its picture of the present matched what was just seen does the
        # graph claim to know what comes next.
        explained &= (graph.render(shade=True) - self.previous) ** 2 < 0.002
        picture = np.where(explained, graph.expect(shade=True), self.previous)
        dx, dy = (int(round(v)) for v in self.velocity)
        moved = _shift(picture, dx, dy)
        # Where the camera brings in what was not seen, the last frame stands in.
        seen = _shift(np.ones_like(picture), dx, dy) > 0
        return np.where(seen, moved, self.previous).astype(np.float32)

    def to_json(self) -> dict:
        data = self.graph.to_json()
        data["camera"] = {
            "moved": [round(float(v), 1) for v in self.travelled],
            "velocity": [round(float(-v), 2) for v in self.velocity],
        }
        return data
