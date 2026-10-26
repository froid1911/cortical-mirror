"""Reproduce the reviewer's check: pan a texture N px per frame, read off the motion.

A predictor that only repeats the last frame reports ~0 px. The texture has a
natural-image 1/f spectrum so that motion is visible at every scale.
"""

import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F

from cortical_mirror.engines import CorticalEngine, resolve_device


def texture(height: int, width: int, seed: int = 0) -> np.ndarray:
    generator = torch.Generator().manual_seed(seed)
    image = torch.zeros(3, height, width)
    for scale in (1, 2, 4, 8, 16, 32):
        noise = torch.rand(1, 3, height // scale, width // scale, generator=generator)
        image += scale * F.interpolate(noise, size=(height, width), mode="bilinear")[0]
    image = (image - image.min()) / (image.max() - image.min())
    return (image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--speed", type=int, default=3, help="pixels per frame, rightward")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    engine = CorticalEngine.from_checkpoint(args.checkpoint, device=resolve_device(args.device))
    width, height = engine.width, engine.height
    canvas = texture(height, width + args.speed * args.frames)
    span = canvas.shape[1]
    frames = [
        canvas[:, span - width - args.speed * t : span - args.speed * t] for t in range(args.frames)
    ]
    margin = 3 * args.speed + 8
    pending, model_errors, copy_errors, motion = None, [], [], []
    for index, frame in enumerate(frames):
        result = engine.predict(np.ascontiguousarray(frame))
        if pending is not None and index >= 3:

            def mae(a, b):
                return float(
                    np.mean(np.abs(a[:, margin:-margin].astype(np.float32) - b[:, margin:-margin]))
                )

            model_errors.append(mae(pending, frame))
            copy_errors.append(mae(frames[index - 1], frame))
        if index >= 2:
            motion.append(float(np.median(engine.last_flow[8:-8, margin:-margin, 0])))
        pending = result.image
    report = {
        "true_speed_px": args.speed,
        "predicted_speed_px": float(np.median(motion)),
        "model_mae": float(np.mean(model_errors)) / 255,
        "copy_mae": float(np.mean(copy_errors)) / 255,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
