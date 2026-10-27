"""The innate front end: what a newborn's retina and thalamus already do.

Luminance passes through centre-surround (difference-of-Gaussians) filtering and
divisive local contrast normalisation, then splits into ON and OFF channels.
Optionally, lagged channels repeat the previous frame's ON/OFF code, like the
lagged cells of the lateral geniculate nucleus. Nothing here is learned.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.nn import functional as F


def _gaussian(sigma: float, size: int = 9) -> torch.Tensor:
    axis = torch.arange(size, dtype=torch.float32) - size // 2
    weights = torch.exp(-(axis**2) / (2 * sigma**2))
    return weights / weights.sum()


def frame_to_gray(frame: np.ndarray, width: int, height: int, device) -> torch.Tensor:
    """Any camera frame (BGR or gray, any size) as luminance ``[1, 1, H, W]`` in [0, 1]."""

    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.shape != (height, width):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    gray = torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32) / 255.0)
    return gray[None, None].to(device)


class Retina:
    def __init__(self, width: int = 128, height: int = 96, lagged: bool = False, device="cpu"):
        self.width, self.height, self.lagged = width, height, lagged
        self.device = torch.device(device)
        centre, surround = _gaussian(1.0), _gaussian(2.0)
        kernel = torch.outer(centre, centre) - torch.outer(surround, surround)
        self._kernel = kernel[None, None].to(self.device)
        self.channels = 4 if lagged else 2
        self.reset()

    def reset(self) -> None:
        self._previous: torch.Tensor | None = None

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Luminance ``[B, 1, H, W]`` to the ON/OFF code ``[B, C, H, W]``."""

        # Replicate the border so the edge of the image is not seen as an edge.
        padded = F.pad(frames.to(self.device), (4, 4, 4, 4), mode="replicate")
        contrast = F.conv2d(padded, self._kernel)
        scale = F.avg_pool2d(F.pad(contrast.abs(), (7, 7, 7, 7), mode="replicate"), 15, 1) + 0.02
        contrast = (contrast / scale).clamp(-3.0, 3.0) / 3.0
        code = torch.cat((F.relu(contrast), F.relu(-contrast)), dim=1)
        if not self.lagged:
            return code
        previous = self._previous if self._previous is not None else torch.zeros_like(code)
        self._previous = code
        return torch.cat((code, previous), dim=1)
