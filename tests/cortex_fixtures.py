"""Natural-looking (1/f) test stimuli for the cortex tests."""

import torch
from torch.nn import functional as F


def texture(height: int, width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    image = torch.zeros(1, 1, height, width)
    for scale in (1, 2, 4, 8, 16):
        noise = torch.rand(
            1, 1, max(2, height // scale), max(2, width // scale), generator=generator
        )
        image += scale * F.interpolate(
            noise, size=(height, width), mode="bilinear", align_corners=False
        )
    image -= image.min()
    return image / image.max()


def pan(
    frames: int, speed: int, height: int = 48, width: int = 64, seed: int = 0
) -> list[torch.Tensor]:
    """Content moving right by ``speed`` pixels per frame."""

    canvas = texture(height, width + speed * frames, seed)
    span = canvas.shape[-1]
    return [canvas[..., span - width - speed * t : span - speed * t] for t in range(frames)]


def scenes(
    frames: int, height: int = 48, width: int = 64, scene_length: int = 60
) -> list[torch.Tensor]:
    """A stream of panning scenes, switching to a new texture and direction every scene."""

    out = []
    for index in range(0, frames, scene_length):
        clip = pan(scene_length, 1 + index // scene_length % 3, height, width, seed=index)
        if (index // scene_length) % 2:
            clip = [frame.flip(-1) for frame in clip]
        out.extend(clip)
    return out[:frames]
