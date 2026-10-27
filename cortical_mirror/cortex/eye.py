"""An eye that moves: a fovea inside a larger view, saccades driven by surprise.

The cortex sees only the foveal window. A coarse peripheral pathway supplies a
surprise map of the whole view. Every few frames the eye makes one fixed-size
saccade in one of eight directions: toward where surprise lies, away from places
it has just looked at (inhibition of return). The chosen command is returned so
the cortex can be told what its own eye is about to do (the efference copy).
Where to look is not built in; only the reflex "look toward surprise" is.
"""

from __future__ import annotations

import math

import torch

# Command 0 keeps the eye still; 1-8 are compass steps, (dx, dy) with y pointing down.
COMMANDS = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]


class Eye:
    def __init__(
        self,
        world_width: int = 256,
        world_height: int = 192,
        fovea: tuple[int, int] = (128, 96),
        saccade_px: int = 8,
        period_frames: int = 10,
        inhibition: float = 0.7,
        memory: int = 8,
    ):
        self.world = (world_width, world_height)
        self.fovea = fovea
        self.saccade_px = saccade_px
        self.period_frames = period_frames
        self.inhibition = inhibition
        self.memory = memory
        self.reset()

    def reset(self) -> None:
        self.gaze = (self.world[0] // 2, self.world[1] // 2)
        self.history: list[tuple[int, int]] = [self.gaze]
        self._frame = 0

    def _inside(self, x: int, y: int) -> bool:
        half_w, half_h = self.fovea[0] // 2, self.fovea[1] // 2
        return half_w <= x <= self.world[0] - half_w and half_h <= y <= self.world[1] - half_h

    def _score(self, target: tuple[int, int], surprise: torch.Tensor) -> float:
        """Surprise weighted by closeness to the target, discounted near recent fixations."""

        height, width = surprise.shape[-2:]
        yy, xx = torch.meshgrid(
            (torch.arange(height) + 0.5) * self.world[1] / height,
            (torch.arange(width) + 0.5) * self.world[0] / width,
            indexing="ij",
        )
        reach = self.world[0] / 4
        weight = torch.exp(-((xx - target[0]) ** 2 + (yy - target[1]) ** 2) / (2 * reach**2))
        value = float((surprise[0, 0].cpu() * weight).sum() / weight.sum())
        # Earlier fixations only: every candidate is one saccade from the current one,
        # so discounting near it would favour some directions over others.
        radius = self.saccade_px / 2
        for x, y in self.history[-self.memory - 1 : -1]:
            closeness = math.exp(-((target[0] - x) ** 2 + (target[1] - y) ** 2) / (2 * radius**2))
            value *= 1.0 - self.inhibition * closeness
        return value

    def look(self, world: torch.Tensor, surprise: torch.Tensor | None) -> tuple[torch.Tensor, int]:
        """Possibly saccade, then return the foveal window ``[1, 1, h, w]`` and the command."""

        self._frame += 1
        command = 0
        if surprise is not None and self._frame % self.period_frames == 0:
            best, best_score = 0, -1.0
            for index, (dx, dy) in enumerate(COMMANDS[1:], start=1):
                target = (self.gaze[0] + dx * self.saccade_px, self.gaze[1] + dy * self.saccade_px)
                if not self._inside(*target):
                    continue
                score = self._score(target, surprise)
                if score > best_score:
                    best, best_score = index, score
            if best:
                dx, dy = COMMANDS[best]
                self.gaze = (
                    self.gaze[0] + dx * self.saccade_px,
                    self.gaze[1] + dy * self.saccade_px,
                )
                self.history.append(self.gaze)
                command = best
        x, y = self.gaze
        half_w, half_h = self.fovea[0] // 2, self.fovea[1] // 2
        return world[..., y - half_h : y + half_h, x - half_w : x + half_w], command
