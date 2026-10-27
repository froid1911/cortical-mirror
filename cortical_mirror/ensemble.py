"""Precision-weighted routing driven only by errors from predictions already due.

In predictive coding, precision is the expected reliability of a prediction error
and decides how much each signal is trusted (Feldman & Friston, 2010). Here each
image region estimates the precision of every predictor from its recent due errors
and weights the predictors' forecasts accordingly.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .engines import EnginePrediction, PredictiveEngine, _resize

MEAN_OF_CHANNELS = np.full((1, 3), 1.0 / 3.0, dtype=np.float32)


class AdaptiveEngine:
    def __init__(
        self,
        engines: dict[str, PredictiveEngine],
        temperature: float = 0.02,
        adaptation: float = 0.15,
        patch_size: int = 16,
    ):
        if not engines:
            raise ValueError("adaptive engine needs at least one expert")
        if temperature <= 0 or not 0 < adaptation <= 1:
            raise ValueError("temperature must be positive and adaptation in (0, 1]")
        self.engines = engines
        first = next(iter(engines.values()))
        self.width, self.height = first.width, first.height
        if any(
            (engine.width, engine.height) != (self.width, self.height)
            for engine in engines.values()
        ):
            raise ValueError("all experts must use the same frame dimensions")
        self.temperature = temperature
        self.adaptation = adaptation
        self.patch_size = patch_size
        self.names = tuple(engines)
        self.last_weights: np.ndarray | None = None
        self.reset()

    def reset(self) -> None:
        for engine in self.engines.values():
            engine.reset()
        self.errors: np.ndarray | None = None
        self.pending: np.ndarray | None = None
        self.pending_experts: dict[str, np.ndarray] | None = None
        self.last_weights = None
        self.routing: np.ndarray | None = None
        self.last_expert_errors: dict[str, float] = {}

    @property
    def last_levels(self):
        return getattr(self.engines.get("cortical"), "last_levels", [])

    @property
    def learner(self):
        return getattr(self.engines.get("cortical"), "learner", None)

    def rollout(self, first_bgr: np.ndarray, steps: int) -> np.ndarray:
        cortical = self.engines.get("cortical")
        if cortical is None or not hasattr(cortical, "rollout"):
            return first_bgr
        return cortical.rollout(first_bgr, steps)

    def _grid(self) -> tuple[int, int]:
        return max(1, self.width // self.patch_size), max(1, self.height // self.patch_size)

    def _patch_error(self, prediction: np.ndarray, actual: np.ndarray) -> np.ndarray:
        """Mean absolute error of one predictor in every routing patch, in [0, 1]."""

        difference = cv2.absdiff(prediction, actual)
        small = cv2.resize(difference, self._grid(), interpolation=cv2.INTER_AREA)
        return small.astype(np.float32).reshape(*small.shape[:2], -1).mean(axis=2) / 255.0

    def _weights(self) -> np.ndarray:
        """Softmax of negative patch error: low recent error means high precision."""

        width, height = self._grid()
        if self.errors is None:
            return np.full((height, width, len(self.engines)), 1.0 / len(self.engines), np.float32)
        logits = -self.errors / self.temperature
        logits -= logits.max(axis=2, keepdims=True)
        weights = np.exp(logits)
        return (weights / weights.sum(axis=2, keepdims=True)).astype(np.float32)

    def predict(self, frame_bgr: np.ndarray) -> EnginePrediction:
        start = time.perf_counter()
        actual = _resize(frame_bgr, self.width, self.height)
        observed = None
        if self.pending is not None:
            observed = cv2.transform(cv2.absdiff(self.pending, actual), MEAN_OF_CHANNELS)
            observed = observed.astype(np.float32) / 255.0
        if self.pending_experts is not None:
            current_errors = np.stack(
                [self._patch_error(self.pending_experts[name], actual) for name in self.names],
                axis=2,
            )
            self.last_expert_errors = {
                name: float(current_errors[..., index].mean())
                for index, name in enumerate(self.names)
            }
            self.errors = (
                current_errors
                if self.errors is None
                else (1.0 - self.adaptation) * self.errors + self.adaptation * current_errors
            )

        predictions = {name: self.engines[name].predict(actual) for name in self.names}
        small_weights = self._weights()
        weights = cv2.resize(
            small_weights, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )
        if weights.ndim == 2:
            weights = weights[..., None]
        weights /= weights.sum(axis=2, keepdims=True)
        self.routing = self.last_weights = weights
        prediction = np.zeros((self.height, self.width, 3), np.float32)
        for index, name in enumerate(self.names):
            prediction += predictions[name].image.astype(np.float32) * weights[..., index, None]
        if self.errors is None:
            confidence = np.full((self.height, self.width), 0.5, dtype=np.float32)
        else:
            expected = np.sum(small_weights * self.errors, axis=2)
            confidence = cv2.resize(
                np.exp(-10.0 * expected).astype(np.float32),
                (self.width, self.height),
                interpolation=cv2.INTER_LINEAR,
            )
        image = np.rint(np.clip(prediction, 0, 255)).astype(np.uint8)
        self.pending = image
        self.pending_experts = {name: predictions[name].image for name in self.names}
        return EnginePrediction(
            image=image,
            confidence=confidence,
            observed_error=observed,
            inference_ms=(time.perf_counter() - start) * 1000.0,
        )
