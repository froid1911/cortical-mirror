"""Record a GIF of the learned scene: what it sees, the objects it found by learned
motion, and the scene with every object deleted. The source is a real video with a
still camera, or the sprite world (``--source world``).

    python tools/record_learned_video_gif.py --mt checkpoints/mt-adult.pt \
        --links checkpoints/links-adult.pt --source checkpoints/sample.mp4 \
        --frames-dir /tmp/learned-video --output docs/images/learned-video.gif
"""

from __future__ import annotations

import argparse
import subprocess
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
from record_slots_gif import COLOURS, MUTED, SIZE, label, picture  # tools/ on the path

from cortical_mirror.cli import learned_graph
from cortical_mirror.scene.live import SpriteSource, VideoSource

LABELS = {
    "en": {
        "sees": "What it sees",
        "objects": "Learned objects",
        "deleted": "Every object deleted",
        "frame": "frame",
        "world": "sprite world",
        "video": "real video",
        "memory": "background from memory",
    },
    # The German labels avoid umlauts, which OpenCV's fonts cannot draw.
    "de": {
        "sees": "Was es sieht",
        "objects": "Gelernte Objekte",
        "deleted": "Alle Objekte geloescht",
        "frame": "Bild",
        "world": "Sprite-Welt",
        "video": "echtes Video",
        "memory": "Hintergrund aus dem Gedaechtnis",
    },
}


def objects(image: np.ndarray, graph) -> np.ndarray:
    out = picture(image).astype(np.float32) * 0.6
    for i, node in enumerate(graph.nodes):
        mask = cv2.resize(node.mask.astype(np.uint8), SIZE, interpolation=cv2.INTER_NEAREST) > 0
        colour = np.array(COLOURS[i % len(COLOURS)], np.float32)
        out[mask] = 0.45 * out[mask] + 0.55 * colour
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mt", type=Path, required=True)
    parser.add_argument("--links", type=Path, required=True)
    parser.add_argument("--source", required=True, help="a still-camera video, or 'world'")
    parser.add_argument("--language", choices=("en", "de"), default="en")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--skip", type=int, default=2, help="record every n-th frame")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    graph = learned_graph(Namespace(mt=args.mt, links=args.links), args.device)
    world = args.source == "world"
    source = SpriteSource(scene_frames=10**9) if world else VideoSource(args.source)
    words = LABELS[args.language]
    args.frames_dir.mkdir(parents=True, exist_ok=True)
    gutter, width = 8, SIZE[0]
    for t in range(args.frames):
        image = source.read()
        graph.observe(image)
        if t % args.skip:
            continue
        empty = graph.edit(delete={n.id for n in graph.nodes})
        canvas = np.zeros((SIZE[1] + 56, 3 * width + 2 * gutter, 3), np.uint8)
        for i, panel in enumerate((picture(image), objects(image, graph), picture(empty))):
            canvas[: SIZE[1], i * (width + gutter) : i * (width + gutter) + width] = panel
        y = SIZE[1] + 22
        label(canvas, words["sees"], 0, y)
        label(canvas, words["objects"], width + gutter, y)
        label(canvas, words["deleted"], 2 * (width + gutter), y)
        where = words["world"] if world else words["video"]
        label(canvas, f"{words['frame']} {t}, {where}", 0, y + 20, MUTED, 0.4)
        label(canvas, words["memory"], 2 * (width + gutter), y + 20, MUTED, 0.4)
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
