"""Pretrained DMVFN baseline with the same causal interface as the cortical engine."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .dmvfn_impl import DMVFN
from .engines import EnginePrediction, _resize

CHECKPOINTS = {
    "city": "1jILbS8Gm4E5Xx4tDCPZh_7rId0eo8r9W",
    "kitti": "1WrV30prRiS4hWOQBnVPUxdaTlp9XxmVK",
    "vimeo": "14_xQ3Yl3mO89hr28hbcQW3h63lLrcYY0",
}


class DMVFNEngine:
    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cuda",
        width: int = 256,
        height: int = 192,
        mixed_precision: bool = True,
    ):
        self.device = torch.device(device)
        self.width, self.height = int(width), int(height)
        self.mixed_precision = mixed_precision and self.device.type == "cuda"
        self.model = DMVFN()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.previous: torch.Tensor | None = None
        self.pending: torch.Tensor | None = None

    def reset(self) -> None:
        self.previous = None
        self.pending = None

    def _tensor(self, frame_bgr: np.ndarray) -> torch.Tensor:
        frame = _resize(frame_bgr, self.width, self.height)
        array = np.ascontiguousarray(frame.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        return torch.from_numpy(array).to(self.device, non_blocking=True)

    def predict(self, frame_bgr: np.ndarray) -> EnginePrediction:
        start = time.perf_counter()
        current = self._tensor(frame_bgr)
        observed = (
            (current - self.pending).abs().mean(dim=1).detach()
            if self.pending is not None
            else None
        )
        if self.previous is None:
            prediction = current.clone()
        else:
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.mixed_precision,
                ),
            ):
                merged = self.model(
                    torch.cat((self.previous, current), dim=1), training=False, deterministic=True
                )
                prediction = merged[-1] if merged else self.previous.clone()
        self.previous = current.detach()
        self.pending = prediction.detach()
        image = prediction[0].float().permute(1, 2, 0).cpu().numpy()
        observed_np = observed[0, 0].float().cpu().numpy() if observed is not None else None
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        elapsed = (time.perf_counter() - start) * 1000.0
        return EnginePrediction(
            image=np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8),
            confidence=np.full((self.height, self.width), np.nan, dtype=np.float32),
            observed_error=observed_np,
            inference_ms=elapsed,
        )
