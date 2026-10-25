"""Online expert routing driven only by errors from predictions already due."""

from __future__ import annotations

import time

import cv2
import numpy as np

from .engines import EnginePrediction, PredictiveEngine, _resize


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

    def _weights(self) -> np.ndarray:
        if self.errors is None:
            self.last_weights = np.full(
                (self.height, self.width, len(self.engines)),
                1.0 / len(self.engines),
                dtype=np.float32,
            )
        else:
            logits = -self.errors / self.temperature
            logits -= logits.max(axis=2, keepdims=True)
            weights = np.exp(logits)
            self.last_weights = weights / weights.sum(axis=2, keepdims=True)
        return self.last_weights

    def predict(self, frame_bgr: np.ndarray) -> EnginePrediction:
        start = time.perf_counter()
        actual = _resize(frame_bgr, self.width, self.height)
        observed = (
            np.mean(np.abs(actual.astype(np.float32) - self.pending), axis=2) / 255.0
            if self.pending is not None
            else None
        )
        if self.pending_experts is not None:
            current_errors = np.stack(
                [
                    np.mean(
                        np.abs(
                            self.pending_experts[name].astype(np.float32)
                            - actual.astype(np.float32)
                        ),
                        axis=2,
                    )
                    / 255.0
                    for name in self.names
                ],
                axis=2,
            )
            self.errors = (
                current_errors
                if self.errors is None
                else (1.0 - self.adaptation) * self.errors + self.adaptation * current_errors
            )

        predictions = {name: self.engines[name].predict(actual) for name in self.names}
        weights = self._weights()
        small_size = (max(1, self.width // self.patch_size), max(1, self.height // self.patch_size))
        small_weights = cv2.resize(weights, small_size, interpolation=cv2.INTER_AREA)
        weights = cv2.resize(
            small_weights, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )
        if weights.ndim == 2:
            weights = weights[..., None]
        weights /= weights.sum(axis=2, keepdims=True)
        prediction = np.sum(
            np.stack([predictions[name].image.astype(np.float32) for name in self.names], axis=2)
            * weights[..., None],
            axis=2,
        )
        confidence = (
            np.full((self.height, self.width), 0.5, dtype=np.float32)
            if self.errors is None
            else np.exp(-10.0 * np.sum(weights * self.errors, axis=2)).astype(np.float32)
        )
        self.pending = prediction
        self.pending_experts = {name: predictions[name].image for name in self.names}
        return EnginePrediction(
            image=np.rint(np.clip(prediction, 0, 255)).astype(np.uint8),
            confidence=confidence,
            observed_error=observed,
            inference_ms=(time.perf_counter() - start) * 1000.0,
        )
