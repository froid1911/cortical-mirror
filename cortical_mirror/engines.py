"""Inference adapters. Every call consumes frame t and returns a forecast for t+1."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import torch

from .model import CorticalFlowNet, ModelState


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

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str = "cuda",
        mixed_precision: bool = True,
    ) -> CorticalEngine:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
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

    def _tensor(self, frame_bgr: np.ndarray) -> torch.Tensor:
        frame = _resize(frame_bgr, self.width, self.height)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        array = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        return torch.from_numpy(array).to(self.device, non_blocking=True)

    def predict(self, frame_bgr: np.ndarray) -> EnginePrediction:
        start = time.perf_counter()
        tensor = self._tensor(frame_bgr)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.mixed_precision)
        else:
            autocast = torch.autocast("cpu", enabled=False)
        with torch.inference_mode(), autocast:
            output = self.model(tensor, self.state)
            self.state = output.state.detached()
        prediction = output.prediction[0].float().permute(1, 2, 0).cpu().numpy()
        confidence = output.state.confidence[0, 0].float().cpu().numpy()
        observed = (
            output.observed_error[0, 0].float().cpu().numpy()
            if output.observed_error is not None
            else None
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = (time.perf_counter() - start) * 1000.0
        bgr = cv2.cvtColor(
            np.rint(np.clip(prediction, 0.0, 1.0) * 255.0).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        return EnginePrediction(
            image=bgr,
            confidence=confidence,
            observed_error=observed,
            inference_ms=elapsed,
        )
