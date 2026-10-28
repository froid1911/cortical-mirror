"""How much could any motion model gain over repeating the last frame on real video?

For each model of how frame t becomes frame t+1, two numbers, both as the mean
squared error of the predicted t+1 over that of repeating frame t:

* oracle: the model fitted to frame t+1 itself (an upper bound on what it can do);
* extrapolated: the model fitted to t-1 -> t and applied again to t (what a
  predictor that assumes motion continues can actually do);
* damped: the same, carrying only a fraction (0.25, 0.5, 0.75) of that motion
  forward;
* smoothed: the motion averaged over the last three frame pairs, carried forward
  at half strength.

Models, from rigid to deformable: one global shift; a translation per 32 px
block; an affine warp per block; a dense smooth flow field (Farneback). This is
a development measurement for v0.6 (validation split), not a protocol.
"""

import argparse
import functools
import json
import random
from pathlib import Path

import cv2
import numpy as np

from cortical_mirror.data import split_videos
from cortical_mirror.scene.graph import _as_gray
from cortical_mirror.scene.phenomena import seed_map

BLOCK = 32
SIZE = (256, 192)


def dense_flow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Flow from a to b, [H, W, 2] (where each pixel of a went)."""

    u8 = [np.rint(x * 255).astype(np.uint8) for x in (a, b)]
    return cv2.calcOpticalFlowFarneback(u8[0], u8[1], None, 0.5, 3, 15, 3, 5, 1.2, 0)


def warp(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Move the image along the flow: the pixel at p goes to p + flow(p),
    approximated by sampling at p - flow(p)."""

    h, w = image.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    return cv2.remap(
        image,
        xx - flow[..., 0],
        yy - flow[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def global_shift(flow: np.ndarray) -> np.ndarray:
    return np.broadcast_to(np.median(flow.reshape(-1, 2), axis=0), flow.shape).copy()


def per_block(flow: np.ndarray, affine: bool) -> np.ndarray:
    """The flow fitted per block by a translation (its median) or an affine map
    (least squares)."""

    h, w, _ = flow.shape
    out = np.empty_like(flow)
    for y in range(0, h, BLOCK):
        for x in range(0, w, BLOCK):
            part = flow[y : y + BLOCK, x : x + BLOCK]
            if not affine:
                out[y : y + BLOCK, x : x + BLOCK] = np.median(part.reshape(-1, 2), axis=0)
                continue
            yy, xx = np.mgrid[: part.shape[0], : part.shape[1]]
            design = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], 1)
            fitted = design @ np.linalg.lstsq(design, part.reshape(-1, 2), rcond=None)[0]
            out[y : y + BLOCK, x : x + BLOCK] = fitted.reshape(part.shape)
    return out


MODELS = {
    "global_shift": global_shift,
    "block_translation": functools.partial(per_block, affine=False),
    "block_affine": functools.partial(per_block, affine=True),
    "dense_flow": lambda flow: flow,
}


def score(now: np.ndarray, after: np.ndarray, flow: np.ndarray) -> float:
    """Mean squared error of the frame moved along a flow, against the next frame."""

    return float(np.mean((warp(now, flow) - after) ** 2))


def clip_errors(path: Path, frames: int, scored_from: int) -> dict:
    capture = cv2.VideoCapture(str(path))
    images = []
    for _ in range(frames):
        ok, frame = capture.read()
        if not ok:
            break
        images.append(_as_gray(frame, *SIZE))
    capture.release()
    kinds = ("oracle", "extrapolated", "damped_0.25", "damped_0.5", "damped_0.75", "smoothed")
    totals = {"repeat": 0.0} | {f"{m}_{k}": 0.0 for m in MODELS for k in kinds}
    for t in range(max(scored_from, 3), len(images) - 1):
        before, now, after = images[t - 1], images[t], images[t + 1]
        totals["repeat"] += float(np.mean((now - after) ** 2))
        ahead, behind = dense_flow(now, after), dense_flow(before, now)
        history = [
            behind,
            dense_flow(images[t - 2], before),
            dense_flow(images[t - 3], images[t - 2]),
        ]
        for name, model in MODELS.items():
            fitted = model(behind)
            totals[f"{name}_oracle"] += score(now, after, model(ahead))
            totals[f"{name}_extrapolated"] += score(now, after, fitted)
            for fraction in (0.25, 0.5, 0.75):
                totals[f"{name}_damped_{fraction}"] += score(now, after, fraction * fitted)
            smoothed = 0.5 * model(np.mean(history, axis=0))
            totals[f"{name}_smoothed"] += score(now, after, smoothed)
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--videos", type=int, default=20)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--scored-from", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    videos = sorted(getattr(split_videos(args.data), args.split))
    chosen = random.Random(0).sample(videos, min(args.videos, len(videos)))
    clips = seed_map(
        functools.partial(clip_errors, frames=args.frames, scored_from=args.scored_from), chosen
    )
    totals = {key: sum(c[key] for c in clips) for key in clips[0]}
    report = {
        "split": args.split,
        "videos": [v.name for v in chosen],
        "ratio_to_repeating": {
            key: value / totals["repeat"] for key, value in totals.items() if key != "repeat"
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["ratio_to_repeating"], indent=1))


if __name__ == "__main__":
    main()
