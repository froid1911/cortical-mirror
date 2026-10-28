"""Record the README's scene-graph GIF: what it sees beside what it explains.

Plays the sprite world through the same live state as `cortical-mirror scene`.
Partway through, one object is deleted and later another is moved, as a click
and a drag on the page would do. ffmpeg turns the frames into the GIF:

    python tools/record_scene_gif.py --adult checkpoints/scene-adult.pt \
        --frames-dir /tmp/scene-gif --output docs/images/scene-graph.gif
"""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

from cortical_mirror.scene.graph import SceneGraph
from cortical_mirror.scene.live import SceneLive, SpriteSource

SCALE = 1.15
EXPECT, SURPRISE = (255, 184, 143), (71, 179, 255)
TEXT, MUTED = (235, 235, 235), (150, 150, 150)


def panel(image: np.ndarray) -> np.ndarray:
    gray = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    size = (int(256 * SCALE), int(192 * SCALE))
    return cv2.cvtColor(cv2.resize(gray, size, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)


def rings(canvas: np.ndarray, scene: dict) -> None:
    deleted = set(scene["edits"]["deleted"])
    for node in scene["nodes"]:
        moved = scene["edits"]["moved"].get(str(node["id"]), [0, 0])
        x, y = node["x"] + moved[0], node["y"] + moved[1]
        centre = (int(x * SCALE), int(y * SCALE))
        radius = int(max(6, math.sqrt(node["size"] / math.pi)) * SCALE)
        colour = SURPRISE if node["id"] in deleted else EXPECT
        if node["state"] == "seen" or node["id"] in deleted:
            cv2.circle(canvas, centre, radius, colour, 1, cv2.LINE_AA)
        else:
            for angle in range(0, 360, 30):
                cv2.ellipse(
                    canvas, centre, (radius, radius), 0, angle, angle + 15, colour, 1, cv2.LINE_AA
                )
        cv2.putText(
            canvas,
            str(node["id"]),
            (centre[0] + radius - 2, centre[1] - radius + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            TEXT,
            1,
            cv2.LINE_AA,
        )


def put(canvas, text, x, y, scale, colour):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adult", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=420)
    parser.add_argument("--every", type=int, default=2)
    args = parser.parse_args()

    state = SceneLive(SpriteSource(seed=5), SceneGraph.load(args.adult))
    args.frames_dir.mkdir(parents=True, exist_ok=True)
    written, caption = 0, ""
    for t in range(args.frames):
        state.step()
        seen_nodes = [n for n in state.scene()["nodes"] if n["state"] == "seen"]
        if t == 160 and seen_nodes:
            target = seen_nodes[0]["id"]
            state.control({"delete": target})
            caption = f"{target} deleted: the background behind it, from memory"
        if t == 300 and len(seen_nodes) > 1:
            target = next(n["id"] for n in seen_nodes if n["id"] not in state.deleted)
            state.control({"move": target, "by": [0, 40]})
            caption = f"{target} moved: its whole shape, somewhere else"
        if t % args.every:
            continue
        scene = state.scene()
        seen = panel(state.source.world.render())
        explained = panel(state.explained)
        rings(explained, scene)
        gap = 12
        width = seen.shape[1] * 2 + gap
        canvas = np.zeros((seen.shape[0] + 56, width, 3), np.uint8)
        canvas[: seen.shape[0], : seen.shape[1]] = seen
        canvas[: seen.shape[0], seen.shape[1] + gap :] = explained
        put(canvas, "What it sees", 0, seen.shape[0] + 20, 0.55, TEXT)
        put(canvas, "What it explains", seen.shape[1] + gap, seen.shape[0] + 20, 0.55, TEXT)
        put(
            canvas,
            caption or "dashed: believed, not seen",
            seen.shape[1] + gap,
            seen.shape[0] + 40,
            0.42,
            MUTED,
        )
        cv2.imwrite(str(args.frames_dir / f"{written:04d}.png"), canvas)
        written += 1

    palette = (
        "split[a][b];[a]palettegen=max_colors=48:stats_mode=full[p];[b][p]paletteuse=dither=none"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "12",
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
