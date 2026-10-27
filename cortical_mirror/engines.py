"""Inference adapters. Every call consumes frame t and returns a forecast for t+1."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from pathlib import Path, PosixPath, WindowsPath
from typing import Protocol

import cv2
import numpy as np
import torch

from .model import CorticalFlowNet, ModelState
from .online import OnlineLearner


def resolve_device(device: str) -> str:
    """Map ``auto`` to CUDA when it is usable and to the CPU otherwise."""

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; use --device auto or --device cpu")
    return device


def load_checkpoint(path: str | Path) -> dict:
    """Load model data without permitting arbitrary checkpoint code execution."""

    with torch.serialization.safe_globals([PosixPath, WindowsPath]):
        return torch.load(path, map_location="cpu", weights_only=True)


@dataclass
class EnginePrediction:
    image: np.ndarray
    confidence: np.ndarray
    observed_error: np.ndarray | None
    inference_ms: float


class PredictiveEngine(Protocol):
    width: int
    height: int

    def reset(self) -> None: ...

    def predict(self, frame_bgr: np.ndarray) -> EnginePrediction: ...


def _resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[:2] == (height, width):
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


class CopyEngine:
    """Persistence baseline: the next frame is predicted to equal the current frame."""

    def __init__(self, width: int = 256, height: int = 192):
        self.width = width
        self.height = height

    def reset(self) -> None:
        return None

    def predict(self, frame_bgr: np.ndarray) -> EnginePrediction:
        start = time.perf_counter()
        frame = _resize(frame_bgr, self.width, self.height)
        elapsed = (time.perf_counter() - start) * 1000.0
        return EnginePrediction(
            image=frame.copy(),
            confidence=np.ones((self.height, self.width), np.float32),
            observed_error=None,
            inference_ms=elapsed,
        )


class CorticalEngine:
    def __init__(
        self,
        model: CorticalFlowNet,
        device: str = "cuda",
        width: int = 256,
        height: int = 192,
        mixed_precision: bool = True,
    ):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.width = int(width)
        self.height = int(height)
        self.mixed_precision = mixed_precision and self.device.type == "cuda"
        self.state: ModelState | None = None
        self._previous: torch.Tensor | None = None
        self._reset_requested = False
        self.learner: OnlineLearner | None = None
        self.last_flow: np.ndarray | None = None
        self.last_levels: list[tuple[np.ndarray, np.ndarray]] = []

    def enable_learning(self, **options) -> OnlineLearner:
        """Keep training on the incoming stream in a background thread."""

        self.learner = OnlineLearner(self.model, **options)
        self.learner.on_rollback = self.request_reset
        self.learner.start()
        return self.learner

    @classmethod
    def fresh(
        cls,
        device: str = "cuda",
        width: int = 256,
        height: int = 192,
        mixed_precision: bool = True,
        **model_options,
    ) -> CorticalEngine:
        """An untrained hierarchy: exact persistence until it learns from the camera."""

        return cls(
            CorticalFlowNet(**model_options),
            device=device,
            width=width,
            height=height,
            mixed_precision=mixed_precision,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str = "cuda",
        mixed_precision: bool = True,
    ) -> CorticalEngine:
        checkpoint = load_checkpoint(path)
        model = CorticalFlowNet(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["state_dict"])
        width, height = checkpoint["image_size"]
        return cls(
            model,
            device=device,
            width=width,
            height=height,
            mixed_precision=mixed_precision,
        )

    def reset(self) -> None:
        self.state = None
        self._previous = None
        self._reset_requested = False

    def request_reset(self) -> None:
        """Ask for a reset from another thread; it happens before the next frame."""

        self._reset_requested = True

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.mixed_precision)
        return contextlib.nullcontext()

    def _weights(self):
        return self.learner.lock if self.learner is not None else contextlib.nullcontext()

    @staticmethod
    def _bgr(prediction: torch.Tensor) -> np.ndarray:
        rgb = prediction[0].float().permute(1, 2, 0).cpu().numpy()
        return cv2.cvtColor(
            np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR
        )

    def rollout(self, first_bgr: np.ndarray, steps: int) -> np.ndarray:
        """Imagine ``steps`` further frames by feeding predictions back as input.

        The committed state is untouched; the next real frame resumes from it.
        """

        if self.state is None or steps <= 0:
            return first_bgr
        state, image = self.state, self._tensor(first_bgr)
        with torch.inference_mode(), self._autocast(), self._weights():
            for _ in range(steps):
                output = self.model(image, state)
                state, image = output.state, output.prediction.float()
        return self._bgr(image)

    def _tensor(self, frame_bgr: np.ndarray) -> torch.Tensor:
        frame = _resize(frame_bgr, self.width, self.height)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        array = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        return torch.from_numpy(array).to(self.device, non_blocking=True)

    def predict(self, frame_bgr: np.ndarray) -> EnginePrediction:
        start = time.perf_counter()
        if self._reset_requested:
            self.reset()
        tensor = self._tensor(frame_bgr)
        if self.learner is not None:
            self.learner.observe(tensor)
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        with torch.inference_mode(), self._autocast(), self._weights():
            output = self.model(tensor, self.state)
            self.state = output.state.detached()
        previous = self._previous
        if self.learner is not None and output.has_ground_truth and previous is not None:
            self.learner.observe_outcome(
                float(output.observed_error.mean()),
                float((tensor - previous).abs().mean()),
            )
        self._previous = tensor
        self.last_flow = output.flow[0].float().permute(1, 2, 0).cpu().numpy()
        # Per level: what it expects next, and how surprised it was by this frame.
        self.last_levels = [
            (
                expected[0].float().permute(1, 2, 0).cpu().numpy(),
                error[0].float().abs().sum(dim=0).cpu().numpy() / 3.0,
            )
            for expected, error in zip(output.state.predicted, output.errors, strict=True)
        ]
        confidence = output.state.confidence[0, 0].float().cpu().numpy()
        observed = (
            output.observed_error[0, 0].float().cpu().numpy()
            if output.observed_error is not None
            else None
        )
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        bgr = self._bgr(output.prediction)
        elapsed = (time.perf_counter() - start) * 1000.0
        return EnginePrediction(
            image=bgr,
            confidence=confidence,
            observed_error=observed,
            inference_ms=elapsed,
        )
