"""Record the README's twin-mirror GIF: a newborn learning beside the adult.

Plays a clip (looping) through the same twins as `cortical-mirror twins` and
writes PNG frames, sampled ever more sparsely so the GIF shows the newborn's
first seconds in real time and then its first minutes sped up. ffmpeg turns
them into the GIF:

    python tools/record_twins_gif.py --source sample.mp4 --adult cortex-adult-p1.pt \
        --frames-dir /tmp/twins-gif --output docs/images/twins.gif
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

from cortical_mirror.cortex.engine import CortexEngine
from cortical_mirror.cortex.twin import TwinEngine

WIDTH, COLUMN, GAP = 600, 288, 24
BACKGROUND, TEXT, MUTED = (0, 0, 0), (235, 235, 235), (150, 150, 150)


def experience(frames: int, fps: float) -> str:
    seconds = round(frames / fps)
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60} s"
    return f"{seconds // 3600} h {round(seconds % 3600 / 60)} min"


def sample_points(total: int, count: int) -> list[int]:
    """Frame indices to keep: dense at birth, then geometrically sparser."""

    real_time = count // 3
    dense = list(range(0, real_time * 3, 3))
    sparse = np.geomspace(dense[-1] + 3, total - 1, count - real_time).round().astype(int)
    return sorted(set(dense) | set(sparse.tolist()))


def put(canvas, text, x, y, scale, colour, thickness=1):
    cv2.putText(
        canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA
    )


def column(view: np.ndarray, cells: np.ndarray, title: str, age: str) -> np.ndarray:
    view = cv2.resize(cv2.flip(view, 1), (COLUMN, COLUMN * 3 // 4), interpolation=cv2.INTER_AREA)
    cells = cv2.resize(cells, (COLUMN, COLUMN // 2), interpolation=cv2.INTER_AREA)
    out = np.zeros((view.shape[0] + 46 + cells.shape[0], COLUMN, 3), np.uint8)
    out[: view.shape[0]] = view
    put(out, title, 0, view.shape[0] + 22, 0.62, TEXT, 1)
    put(out, age, 0, view.shape[0] + 40, 0.42, MUTED)
    out[view.shape[0] + 46 :] = cells
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--adult", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total", type=int, default=3000, help="newborn frames to play")
    parser.add_argument("--count", type=int, default=111, help="GIF frames")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    adult = CortexEngine.load(args.adult, device=args.device, learn=True)
    twins = TwinEngine(CortexEngine(device=args.device, learn=True, phase=adult.phase), adult)
    capture = cv2.VideoCapture(str(args.source))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    keep = set(sample_points(args.total, args.count))
    args.frames_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for index in range(args.total):
        ok, frame = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = capture.read()
        twins.predict(frame)
        if index not in keep:
            continue
        views = twins.extra_views()
        left = column(
            views["newborn"],
            views["newborn_cells"],
            "Newborn",
            f"{experience(twins.newborn.experience_frames, fps)} of seeing",
        )
        right = column(
            views["adult"],
            views["adult_cells"],
            "Adult",
            f"{experience(adult.experience_frames, 25)} of video before today",
        )
        canvas = np.zeros((left.shape[0] + 24, WIDTH, 3), np.uint8)
        x = (WIDTH - 2 * COLUMN - GAP) // 2
        canvas[12 : 12 + left.shape[0], x : x + COLUMN] = left
        canvas[12 : 12 + right.shape[0], x + COLUMN + GAP : x + 2 * COLUMN + GAP] = right
        cv2.imwrite(str(args.frames_dir / f"{written:04d}.png"), canvas)
        written += 1

    # Everything is grey: 32 levels without dithering keeps the GIF near 4 MB.
    palette = (
        "split[a][b];[a]palettegen=max_colors=32:stats_mode=full[p];[b][p]paletteuse=dither=none"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "8",
            "-i",
            str(args.frames_dir / "%04d.png"),
            "-filter_complex",
            palette,
            str(args.output),
        ],
        check=True,
    )
    print(f"{written} frames -> {args.output}")


if __name__ == "__main__":
    main()
