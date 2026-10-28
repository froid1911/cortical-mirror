"""What the scene graph sees: the cortex's own code of each frame, or plain brightness.

The cortex front runs a frame through the v0.3 cortex: the retina's normalised
ON/OFF contrast, then V1, settled on it, then V1's learned generative weights
rendering what V1 represents back into retinal contrast. The graph matches
objects in that code: contrast where V1 explains it, largely indifferent to the
overall light. Brightness is kept beside it for the picture and for editing.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from ..cortex.engine import RETINA_SIZE, CortexEngine
from ..cortex.retina import Retina


class LuminanceFront:
    """v0.4's view: the graph matches on brightness."""

    name = "luminance"

    def code(self, image: np.ndarray) -> np.ndarray:
        return image


class CortexFront:
    name = "cortex"

    def __init__(self, adult: str | Path | None = None, iterations: int = 20):
        engine = CortexEngine.load(adult, learn=False) if adult else CortexEngine(learn=False)
        self.hierarchy = engine.hierarchy
        self.retina = Retina(*RETINA_SIZE)
        self.iterations = iterations
        self._step_sizes = False

    @torch.no_grad()
    def code(self, image: np.ndarray) -> np.ndarray:
        """``[H, W]`` brightness in [0, 1] → ``[H, W]`` code in [0, 1] (0.5: no contrast)."""

        height, width = image.shape
        small = cv2.resize(image, RETINA_SIZE, interpolation=cv2.INTER_AREA)
        retinal = self.retina.encode(torch.from_numpy(small)[None, None])
        hierarchy = self.hierarchy
        if not self._step_sizes:
            # The settling step depends only on the weights: estimate it once.
            hierarchy._refresh_step_sizes(retinal)
            self._step_sizes = True
        level = hierarchy.levels[0]
        activity = level.settle(retinal, hierarchy._zeros(retinal)[0], self.iterations)
        rendered = level.render(activity)[0]
        contrast = (rendered[0] - rendered[1]).clamp(-1.0, 1.0).numpy()
        code = 0.5 + 0.5 * cv2.resize(contrast, (width, height), interpolation=cv2.INTER_LINEAR)
        return code.astype(np.float32)


class NormalisedFront:
    """Brightness divided by its surroundings: each pixel as a z-score against the
    mean and spread of its neighbourhood (σ = 24 px).

    This is the retina's divisive normalisation of contrast, applied at the scale of
    surfaces rather than edges. It keeps what an object's surface looks like
    relative to its surroundings, and drops the overall gain and offset of the light.
    """

    name = "normalised"

    def __init__(self, scale: float = 24.0):
        self.scale = scale

    def code(self, image: np.ndarray) -> np.ndarray:
        mean = cv2.GaussianBlur(image, (0, 0), self.scale)
        deviation = image - mean
        spread = np.sqrt(cv2.GaussianBlur(deviation * deviation, (0, 0), self.scale))
        return np.clip(0.5 + 0.15 * deviation / (spread + 0.02), 0.0, 1.0).astype(np.float32)
