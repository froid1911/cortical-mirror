"""The learning cortex behind the mirror's engine interface.

Each camera frame passes through the retina into the hierarchy, which settles,
learns from it (unless frozen), and forecasts the next retinal code. A display
adapter, which is not part of the cortex, turns that forecast into a picture:
the current frame's coarse luminance plus the predicted fine contrast.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch

from ..engines import EnginePrediction
from .hierarchy import Hierarchy, LevelConfig
from .retina import Retina, frame_to_gray

RETINA_SIZE = (128, 96)


def phase_levels(phase: int) -> list[LevelConfig]:
    """Phase 1: V1. From phase 2: V2 above it, predicting V1's activity."""

    levels = [LevelConfig()]
    if phase >= 2:
        # V2 is driven by sparse V1 activity, a far weaker signal than the retina:
        # at V1's threshold it barely fires (rates of 0.0002-0.002 across seeds).
        levels.append(LevelConfig(cells=64, kernel=6, stride=2, padding=2, sparsity=0.01))
    return levels


def render_prediction(
    code_next: torch.Tensor, frame_bgr: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Display adapter: coarse luminance now plus the cortex's predicted contrast."""

    gray = frame_bgr if frame_bgr.ndim == 2 else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    coarse = cv2.GaussianBlur(gray / 255.0, (0, 0), 3.0)
    contrast = (code_next[0, 0] - code_next[0, 1]).float().cpu().numpy()
    contrast = cv2.resize(contrast, (width, height), interpolation=cv2.INTER_LINEAR)
    image = np.clip(coarse + 0.6 * contrast, 0.0, 1.0)
    return cv2.cvtColor(np.rint(image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


class CortexEngine:
    def __init__(
        self,
        device="cpu",
        learn: bool = True,
        width: int = 256,
        height: int = 192,
        phase: int = 1,
        seed: int = 0,
    ):
        self.device = torch.device(device)
        self.learn = learn
        self.width, self.height, self.phase = width, height, phase
        self.retina = Retina(*RETINA_SIZE, lagged=phase >= 2, device=self.device)
        self.hierarchy = Hierarchy(self.retina.channels, phase_levels(phase), self.device, seed)
        self.last_code_error = float("nan")
        self.last_persistence_error = float("nan")
        self.last_stillness_error = float("nan")
        # Frames seen per learning step: 1 live, more when an adult trains on parallel streams.
        self.streams = 1

    @property
    def age_frames(self) -> int:
        return self.hierarchy.age_frames

    @property
    def experience_frames(self) -> int:
        return self.hierarchy.age_frames * self.streams

    def reset(self) -> None:
        self.retina.reset()
        self.hierarchy.reset_state()

    def reset_learning(self) -> None:
        """Make it a newborn again."""

        self.retina.reset()
        self.hierarchy.reset_learning()

    def predict(self, frame_bgr: np.ndarray) -> EnginePrediction:
        start = time.perf_counter()
        gray = frame_to_gray(frame_bgr, *RETINA_SIZE, self.device)
        result = self.hierarchy.step(self.retina.encode(gray), learn=self.learn)
        self.last_code_error = result.code_error
        self.last_persistence_error = result.persistence_error
        self.last_stillness_error = result.stillness_error
        image = render_prediction(result.predicted_next, frame_bgr, self.width, self.height)
        error = result.errors[0].abs().mean(dim=1)[0].cpu().numpy()
        return EnginePrediction(
            image=image,
            confidence=np.ones((self.height, self.width), np.float32),
            observed_error=error,
            inference_ms=(time.perf_counter() - start) * 1000.0,
        )

    def receptive_fields(self, columns: int = 8, scale: int = 6) -> np.ndarray:
        """Every V1 cell's preferred pattern (ON minus OFF weights) as a BGR grid."""

        weights = self.hierarchy.levels[0].W
        fields = (weights[:, 0] - weights[:, 1]).float().cpu()
        low = fields.amin((1, 2), keepdim=True)
        high = fields.amax((1, 2), keepdim=True)
        fields = ((fields - low) / (high - low).clamp_min(1e-6)).numpy()
        cells, size = fields.shape[0], fields.shape[1]
        rows = -(-cells // columns)
        grid = np.full((rows * (size + 1), columns * (size + 1)), 0.5, np.float32)
        for index, field in enumerate(fields):
            r, c = divmod(index, columns)
            grid[r * (size + 1) : r * (size + 1) + size, c * (size + 1) : c * (size + 1) + size] = (
                field
            )
        grid = cv2.resize(grid, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(np.rint(grid * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def save(self, path: str | Path) -> None:
        state = {
            "engine": {"phase": self.phase, "width": self.width, "height": self.height},
            "streams": self.streams,
            "hierarchy": self.hierarchy.state_dict(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path, device="cpu", learn: bool = True) -> CortexEngine:
        state = torch.load(path, map_location="cpu", weights_only=True)
        engine = cls(device=device, learn=learn, **state["engine"])
        engine.hierarchy.load_state_dict(state["hierarchy"])
        engine.streams = int(state.get("streams", 1))
        return engine
