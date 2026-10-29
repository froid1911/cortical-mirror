"""Record a GIF of objects found by learned motion: the picture, MT's velocity, the slots.

The sprite world runs, stops for a while (nothing moves; only the learned links can
hold the objects), then runs again. ffmpeg turns the frames into the GIF:

    python tools/record_slots_gif.py --mt checkpoints/mt-adult-v09.pt \
        --field checkpoints/links-gap4.pt --frames-dir /tmp/slots-gif \
        --output docs/images/slots.gif
"""

from __future__ import annotations

import argparse
import subprocess
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from slots_report import view  # tools/ is on the path when run as a script

from cortical_mirror.cortex.grouping import AssociationField
from cortical_mirror.cortex.mt import MotionCortex
from cortical_mirror.cortex.slots import MotionSlots
from cortical_mirror.scene.world import SpriteWorld

SCALE = 1.15
SIZE = (int(256 * SCALE), int(192 * SCALE))
TEXT, MUTED = (235, 235, 235), (150, 150, 150)
# Slot colours (BGR), the background slot left uncoloured.
COLOURS = [(255, 170, 80), (80, 200, 255), (120, 230, 120), (230, 120, 230), (90, 120, 255)]


def picture(image: np.ndarray) -> np.ndarray:
    gray = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    gray = cv2.resize(gray, SIZE, interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def motion(velocity: np.ndarray) -> np.ndarray:
    """Hue = direction, brightness = speed (1.5 retinal px per frame is full)."""

    vx, vy = velocity
    angle = (np.degrees(np.arctan2(vy, vx)) % 360) / 2
    value = np.clip(np.hypot(vx, vy) / 1.5, 0, 1) * 255
    hsv = np.stack([angle, np.full_like(angle, 220), value], -1).astype(np.uint8)
    return cv2.resize(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), SIZE, interpolation=cv2.INTER_NEAREST)


def slots(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = picture(image).astype(np.float32) * 0.6
    grid = cv2.resize(labels.astype(np.uint8), SIZE, interpolation=cv2.INTER_NEAREST)
    for k in range(1, int(grid.max()) + 1):
        colour = np.array(COLOURS[(k - 1) % len(COLOURS)], np.float32)
        mask = grid == k
        out[mask] = 0.45 * out[mask] + 0.55 * colour
    return np.clip(out, 0, 255).astype(np.uint8)


def label(canvas: np.ndarray, text: str, x: int, y: int, colour=TEXT, size=0.5) -> None:
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, colour, 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mt", type=Path, required=True)
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=905)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--pause", type=int, nargs=2, default=(120, 160))
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cortex = MotionCortex.load(args.mt, device=args.device)
    cortex.fine.smoothing = cortex.coarse.smoothing = 3
    state = torch.load(args.field, map_location="cpu", weights_only=False)
    field = AssociationField(types=state["links"].shape[0], device=args.device)
    field.links = state["links"].to(args.device)
    found = MotionSlots(field=field, device=args.device)
    # MT compares V1 four frames apart, as in protocols 20 and 21.
    world, past = SpriteWorld(seed=args.seed), deque(maxlen=4 + 1)

    args.frames_dir.mkdir(parents=True, exist_ok=True)
    gutter, width = 8, SIZE[0]
    for t in range(args.frames):
        paused = args.pause[0] <= t < args.pause[1]
        if not paused:
            world.step()
        image = world.render()
        past.append(cortex.v1(view(world, cortex.device)))
        if len(past) < 2:
            continue
        velocity = cortex.between(past[0], past[-1])[0] / (len(past) - 1)
        labels = found.step(velocity, past[-1][0]).numpy()

        canvas = np.zeros((SIZE[1] + 56, 3 * width + 2 * gutter, 3), np.uint8)
        panels = (picture(image), motion(velocity.cpu().numpy()), slots(image, labels))
        for i, panel in enumerate(panels):
            canvas[: SIZE[1], i * (width + gutter) : i * (width + gutter) + width] = panel
        y = SIZE[1] + 22
        label(canvas, "Was es sieht", 0, y)
        label(canvas, "Gelernte Bewegung (MT)", width + gutter, y)
        label(canvas, "Objekte (Slots)", 2 * (width + gutter), y)
        label(canvas, "Farbe = Richtung, Helligkeit = Tempo", width + gutter, y + 20, MUTED, 0.4)
        status = "Pause: nichts bewegt sich" if paused else f"Bild {t}"
        label(canvas, status, 2 * (width + gutter), y + 20, MUTED, 0.4)
        cv2.imwrite(str(args.frames_dir / f"{t:04d}.png"), canvas)

    palette = (
        f"fps={args.fps},split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(args.fps),
            "-pattern_type",
            "glob",
            "-i",
            str(args.frames_dir / "*.png"),
            "-filter_complex",
            palette,
            str(args.output),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
