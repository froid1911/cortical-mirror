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

from .eye import EyeGraph, recentre
from .front import LuminanceFront
from .graph import _as_gray
from .motion import PartMotion


class VideoGraph:
    def __init__(
        self,
        size: tuple[int, int] = (256, 192),
        front=None,
        persistence: float = 30.0,
        motion: PartMotion | None | str = "parts",
    ):
        self.size = size
        self.front = front or LuminanceFront()
        # v0.6: the next frame from how the graph's parts move (None: v0.5's expectation).
        self.motion = PartMotion() if motion == "parts" else motion
        self.eye = EyeGraph(size, persistence=persistence, learn_motor=False)
        # The camera's motion, as the shift of the view's content per frame (to the
        # sub-pixel, from the frames themselves), and how far it has moved in all.
        self.velocity = np.zeros(2)
        self.flow = np.zeros(2)
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
        self.flow += 0.5 * (shift - self.flow)
        self.eye.motor[0] = -np.round(shift)
        gaze, before = self.eye.gaze, self.eye.world_shift.copy()
        self.eye.observe(code, 0, shade=gray)
        # In the view, content moved by minus the eye's movement plus the world's.
        content = -(np.array(self.eye.gaze) - np.array(gaze)) + (self.eye.world_shift - before)
        self.velocity += 0.5 * (content - self.velocity)
        self.travelled -= content
        recentre(self.eye)
        if self.motion is not None and self.previous is not None:
            parts = {
                node.id: self.graph._alpha(node) > 0.5
                for node in self.graph.nodes
                if node.state == "seen"
            }
            self.motion.observe(self.previous, gray, parts)
        self.previous, self._previous_code = gray, code

    def expect(self) -> np.ndarray:
        """The next frame, ``[H, W]`` brightness."""

        if self.previous is None:
            return np.zeros(self.size[::-1], np.float32)
        if self.motion is not None:
            return self.motion.predict(self.previous)
        graph = self.graph
        explained = graph.confidence > 0.5
        for node in graph.nodes:
            if node.state == "seen":
                explained |= graph._alpha(node) > 0.5
        # Only where its picture of the present matched what was just seen does the
        # graph claim to know what comes next.
        explained &= (graph.render(shade=True) - self.previous) ** 2 < 0.002
        picture = np.where(explained, graph.expect(shade=True), self.previous)
        # The camera keeps moving as it just did, to the sub-pixel; at the border it
        # brings in what was not seen, and the nearest seen pixels stand in.
        dx, dy = self.flow
        warp = np.float32([[1, 0, dx], [0, 1, dy]])
        moved = cv2.warpAffine(
            picture, warp, self.size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        return moved.astype(np.float32)

    def to_json(self) -> dict:
        data = self.graph.to_json()
        data["camera"] = {
            "moved": [round(float(v), 1) for v in self.travelled],
            "velocity": [round(float(-v), 2) for v in self.velocity],
        }
        return data
