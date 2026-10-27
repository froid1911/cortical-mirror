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
from torch.nn import functional as F

from ..engines import EnginePrediction
from .eye import COMMANDS, Eye
from .hierarchy import Hierarchy, LevelConfig, StepResult
from .retina import Retina, frame_to_gray

RETINA_SIZE = (128, 96)
# Phase 3: the eye looks at a fovea-sized window of a world twice as large, and a
# coarse peripheral pathway watches the whole world.
WORLD_SIZE = (256, 192)
PERIPHERY_SIZE = (64, 48)


def phase_levels(phase: int) -> list[LevelConfig]:
    """Phase 1: V1. From phase 2: V2 above it, predicting V1's activity."""

    levels = [LevelConfig(motor_commands=len(COMMANDS) if phase >= 3 else 1)]
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


def render_world(
    code_next: torch.Tensor, frame_bgr: np.ndarray, gaze: tuple[int, int]
) -> np.ndarray:
    """Display adapter for the eye: coarse world, the fovea's predicted contrast where
    the eye will look next, and a frame around the fovea."""

    width, height = WORLD_SIZE
    gray = frame_bgr if frame_bgr.ndim == 2 else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    image = cv2.GaussianBlur(gray / 255.0, (0, 0), 3.0)
    contrast = (code_next[0, 0] - code_next[0, 1]).float().cpu().numpy()
    x, y = gaze
    half_w, half_h = RETINA_SIZE[0] // 2, RETINA_SIZE[1] // 2
    window = image[y - half_h : y + half_h, x - half_w : x + half_w]
    image[y - half_h : y + half_h, x - half_w : x + half_w] = np.clip(window + 0.6 * contrast, 0, 1)
    out = cv2.cvtColor(np.rint(image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.rectangle(
        out, (x - half_w, y - half_h), (x + half_w - 1, y + half_h - 1), (255, 184, 143), 1
    )
    return out


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
        self.input_size = WORLD_SIZE if phase >= 3 else RETINA_SIZE
        self.periphery_retina = Retina(*PERIPHERY_SIZE, device=self.device)
        self.periphery = (
            Hierarchy(2, [LevelConfig(cells=16)], self.device, seed + 1) if phase >= 3 else None
        )
        self.eyes: list[Eye] = []
        self._periphery_previous: torch.Tensor | None = None
        self.last_surprise: torch.Tensor | None = None
        self.last_command = 0
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
        if self.periphery is not None:
            self.periphery.reset_state()
        self._periphery_previous = None
        for eye in self.eyes:
            eye.reset()

    def reset_learning(self) -> None:
        """Make it a newborn again."""

        self.reset()
        self.hierarchy.reset_learning()
        if self.periphery is not None:
            self.periphery.reset_learning()

    @torch.no_grad()
    def observe(self, world: torch.Tensor) -> StepResult:
        """One frame per stream, ``[B, 1, H, W]`` at ``input_size``."""

        if self.periphery is None:
            return self.hierarchy.step(self.retina.encode(world), learn=self.learn)
        coarse = F.interpolate(world, size=PERIPHERY_SIZE[::-1], mode="area")
        code = self.periphery_retina.encode(coarse)
        # Expected input: what was just seen plus the change the periphery has learned
        # to predict. Untrained, that is no change, so surprise is the raw transient.
        change = self.periphery.predicted_change()
        previous = self._periphery_previous
        if previous is not None and previous.shape == code.shape:
            expected = previous + (change if change is not None else 0.0)
            surprise = (code - expected).abs().mean(dim=1, keepdim=True)
        else:
            surprise = torch.zeros_like(code[:, :1])
        self._periphery_previous = code
        self.periphery.step(code, learn=self.learn)
        self.last_surprise = surprise
        while len(self.eyes) < world.shape[0]:
            self.eyes.append(Eye(*WORLD_SIZE, fovea=RETINA_SIZE))
        fovea = torch.cat([eye.crop(world[i : i + 1]) for i, eye in enumerate(self.eyes)])
        # Decide each eye's next movement, and tell the cortex (the efference copy).
        commands = torch.tensor(
            [eye.decide(surprise[i : i + 1]) for i, eye in enumerate(self.eyes)],
            device=self.device,
        )
        self.last_command = int(commands[0])
        return self.hierarchy.step(self.retina.encode(fovea), learn=self.learn, commands=commands)

    def predict(self, frame_bgr: np.ndarray) -> EnginePrediction:
        start = time.perf_counter()
        gray = frame_to_gray(frame_bgr, *self.input_size, self.device)
        result = self.observe(gray)
        self.last_code_error = result.code_error
        self.last_persistence_error = result.persistence_error
        self.last_stillness_error = result.stillness_error
        if self.periphery is None:
            image = render_prediction(result.predicted_next, frame_bgr, self.width, self.height)
        else:
            image = render_world(result.predicted_next, frame_bgr, self.eyes[0].gaze)
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
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
            "periphery": self.periphery.state_dict() if self.periphery is not None else None,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path, device="cpu", learn: bool = True) -> CortexEngine:
        state = torch.load(path, map_location="cpu", weights_only=True)
        engine = cls(device=device, learn=learn, **state["engine"])
        engine.hierarchy.load_state_dict(state["hierarchy"])
        engine.streams = int(state.get("streams", 1))
        if engine.periphery is not None and state.get("periphery"):
            engine.periphery.load_state_dict(state["periphery"])
        return engine
