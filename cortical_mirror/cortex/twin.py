"""Newborn and Adult: the same cortex, with and without experience, side by side."""

from __future__ import annotations

from collections import deque

import numpy as np

from ..engines import EnginePrediction
from .engine import CortexEngine


class _Surprise:
    """Prediction error relative to the cortex expecting nothing to change, over ~1 s."""

    def __init__(self, frames: int = 30):
        self.model: deque[float] = deque(maxlen=frames)
        self.copy: deque[float] = deque(maxlen=frames)

    def add(self, engine: CortexEngine) -> None:
        if np.isfinite(engine.last_code_error) and np.isfinite(engine.last_stillness_error):
            self.model.append(engine.last_code_error)
            self.copy.append(engine.last_stillness_error)

    def ratio(self) -> float | None:
        total = sum(self.copy)
        return sum(self.model) / total if total > 0 else None


class TwinEngine:
    def __init__(self, newborn: CortexEngine, adult: CortexEngine):
        self.newborn, self.adult = newborn, adult
        self.width, self.height = newborn.width, newborn.height
        self._reset_requested = False
        self._views: dict[str, np.ndarray] = {}
        self._surprise = {"newborn": _Surprise(), "adult": _Surprise()}
        self._learns = (newborn.learn, adult.learn)

    def set_learning(self, enabled: bool) -> None:
        """Pause or resume learning for both, keeping each one's own setting
        (a frozen adult stays frozen). Used while the eyes are closed."""

        self.newborn.learn = enabled and self._learns[0]
        self.adult.learn = enabled and self._learns[1]

    def reset(self) -> None:
        self.newborn.reset()
        self.adult.reset()

    def request_newborn_reset(self) -> None:
        """Called from another thread; the newborn restarts before the next frame."""

        self._reset_requested = True

    def predict(self, frame_bgr: np.ndarray) -> EnginePrediction:
        if self._reset_requested:
            self._reset_requested = False
            self.newborn.reset_learning()
            self._surprise["newborn"] = _Surprise()
        newborn = self.newborn.predict(frame_bgr)
        adult = self.adult.predict(frame_bgr)
        self._surprise["newborn"].add(self.newborn)
        self._surprise["adult"].add(self.adult)
        self._views = {
            "newborn": newborn.image,
            "adult": adult.image,
            "newborn_cells": self.newborn.receptive_fields(),
            "adult_cells": self.adult.receptive_fields(),
        }
        return newborn

    def extra_views(self) -> dict[str, np.ndarray]:
        return self._views

    def extra_metrics(self) -> dict:
        return {
            "mode": "twins",
            "twins": {
                name: {
                    "age_frames": engine.age_frames,
                    "experience_frames": engine.experience_frames,
                    "surprise": self._surprise[name].ratio(),
                }
                for name, engine in (("newborn", self.newborn), ("adult", self.adult))
            },
        }
