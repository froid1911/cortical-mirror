"""Run the scene graph on real video and report how well it explains it.

No ground truth exists for real video, so this is a description, not a protocol:
how well the graph's rendering matches each frame, how well its expectation
predicts the next frame against simply repeating the last one, and how many
objects it believes in.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from cortical_mirror.scene.graph import SceneGraph, _as_gray


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--adult", type=Path)
    parser.add_argument("--frames", type=int, default=750)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    graph = SceneGraph.load(args.adult) if args.adult else SceneGraph()
    capture = cv2.VideoCapture(str(args.source))
    explained, expected, repeated, counts = [], [], [], []
    previous = None
    for _ in range(args.frames):
        ok, frame = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = capture.read()
        image = _as_gray(frame, 256, 192)
        if previous is not None:
            expected.append(float(np.mean((graph.expect() - image) ** 2)))
            repeated.append(float(np.mean((previous - image) ** 2)))
        graph.observe(image)
        explained.append(float(np.mean((graph.render() - image) ** 2)))
        counts.append(len([n for n in graph.nodes if n.state == "seen"]))
        previous = image
    report = {
        "source": args.source.name,
        "frames": args.frames,
        "explained_mse": float(np.mean(explained[100:])),
        "expectation_mse": float(np.mean(expected[100:])),
        "repeat_last_frame_mse": float(np.mean(repeated[100:])),
        "objects_in_view_mean": float(np.mean(counts[100:])),
        "objects_in_view_max": int(np.max(counts)),
    }
    report["expectation_over_repeat"] = report["expectation_mse"] / report["repeat_last_frame_mse"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
