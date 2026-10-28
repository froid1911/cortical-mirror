"""Protocol 15: the scene graph's expectation on UCF101 against repeating the last frame.

Three conditions: v0.4's graph (luminance, no registration), the v0.5 video graph
on luminance, and the v0.5 video graph on the normalised code. Use the validation
split while developing; the test split is the protocol.
"""

import argparse
import functools
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np

from cortical_mirror.data import split_videos
from cortical_mirror.scene.front import NormalisedFront
from cortical_mirror.scene.graph import SceneGraph, _as_gray
from cortical_mirror.scene.phenomena import seed_map
from cortical_mirror.scene.video import VideoGraph


class Plain:
    """v0.4 on video: the scene graph's expectation, no registration."""

    def __init__(self):
        self.graph = SceneGraph()

    def observe(self, frame):
        self.graph.observe(frame)

    def expect(self):
        return self.graph.expect(shade=True)


def registration_normalised():
    return VideoGraph(front=NormalisedFront())


CONDITIONS = {
    "v0.4": Plain,
    "registration_luminance": VideoGraph,
    "registration_normalised": registration_normalised,
}


def clip_errors(path: Path, condition: str, frames: int, scored_from: int):
    model = CONDITIONS[condition]()
    capture = cv2.VideoCapture(str(path))
    expected, repeated, previous = [], [], None
    for t in range(frames):
        ok, frame = capture.read()
        if not ok:
            break
        gray = _as_gray(frame, 256, 192)
        if previous is not None and t >= scored_from:
            expected.append(float(np.mean((model.expect() - gray) ** 2)))
            repeated.append(float(np.mean((previous - gray) ** 2)))
        model.observe(gray)
        previous = gray
    capture.release()
    return sum(expected), sum(repeated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--videos", type=int, default=40)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--scored-from", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    videos = sorted(getattr(split_videos(args.data), args.split))
    chosen = random.Random(0).sample(videos, min(args.videos, len(videos)))
    report = {"split": args.split, "videos": [v.name for v in chosen]}
    for name in CONDITIONS:
        started = time.time()
        clips = seed_map(
            functools.partial(
                clip_errors, condition=name, frames=args.frames, scored_from=args.scored_from
            ),
            chosen,
        )
        totals = [sum(e for e, _ in clips), sum(r for _, r in clips)]
        wins = sum(e < r for e, r in clips)
        report[name] = {
            "ratio": totals[0] / totals[1] if totals[1] else float("nan"),
            "clips_better_than_repeating": wins / len(chosen),
            "seconds": round(time.time() - started, 1),
        }
        print(name, json.dumps(report[name]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
