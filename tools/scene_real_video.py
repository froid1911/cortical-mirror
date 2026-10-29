"""Protocol 15: the scene graph's expectation on UCF101 against repeating the last frame.

Three conditions: v0.4's graph (luminance, no registration), the v0.5 video graph
on luminance, and the v0.5 video graph on the normalised code. Use the validation
split while developing; the test split is the protocol.
"""

import argparse
import functools
import json
import os
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


def registration_luminance():
    return VideoGraph(motion=None)


def registration_normalised():
    """v0.5: camera registration, the normalised code, the graph's own expectation."""

    return VideoGraph(front=NormalisedFront(), motion=None)


def part_motion():
    """v0.6: the next frame from the smoothed affine motion of the graph's parts."""

    return VideoGraph(front=NormalisedFront())


MT_CHECKPOINT = os.environ.get("SCENE_MT")


def learned_motion(silenced: bool = False):
    """v0.7: part motion read from learned MT cells (protocol 17)."""

    from cortical_mirror.cortex.mt import MotionCortex

    cortex = MotionCortex.load(MT_CHECKPOINT)
    cortex.silenced = silenced
    return VideoGraph(front=NormalisedFront(), motion_source="learned", mt=cortex)


def mt_silenced():
    return learned_motion(silenced=True)


LINKS_CHECKPOINT = os.environ.get("SCENE_LINKS")


def learned_scene(silenced: bool = False):
    """v0.13: objects found by learned motion over a remembered background (protocol 26)."""

    from argparse import Namespace

    from cortical_mirror.cli import learned_graph

    graph = learned_graph(Namespace(mt=MT_CHECKPOINT, links=LINKS_CHECKPOINT), "cpu")
    graph.silenced = silenced
    graph.reset()
    return graph


def learned_scene_silenced():
    return learned_scene(silenced=True)


CONDITIONS = {
    "v0.4": Plain,
    "registration_luminance": registration_luminance,
    "registration_normalised": registration_normalised,
    "part_motion": part_motion,
    "learned_motion": learned_motion,
    "mt_silenced": mt_silenced,
    "learned_scene": learned_scene,
    "learned_scene_silenced": learned_scene_silenced,
}


def camera_shift(path: Path, frames: int = 150) -> float:
    """The median global shift between frames (phase correlation), in px of 256 × 192:
    how still the camera is. It looks only at the video, never at a model."""

    capture = cv2.VideoCapture(str(path))
    shifts, previous = [], None
    for _ in range(frames):
        ok, frame = capture.read()
        if not ok:
            break
        gray = _as_gray(frame, 256, 192)
        if previous is not None:
            (dx, dy), _ = cv2.phaseCorrelate(previous, gray)
            shifts.append(float(np.hypot(dx, dy)))
        previous = gray
    capture.release()
    return float(np.median(shifts)) if shifts else float("inf")


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
    parser.add_argument("--seed", type=int, default=0, help="which clips are drawn")
    parser.add_argument(
        "--exclude-seed",
        type=int,
        action="append",
        help="leave out the clips another seed drew (fresh clips); may be repeated",
    )
    parser.add_argument("--conditions", help="comma-separated; default: all")
    parser.add_argument("--mt", type=Path, help="MT adult, for the learned-motion conditions")
    parser.add_argument("--links", type=Path, help="grouping links, for the learned scene")
    parser.add_argument(
        "--still", type=float, help="only clips whose camera moves less (median px per frame)"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.mt:
        # Read by the worker processes, which build their own conditions.
        os.environ["SCENE_MT"] = str(args.mt)
        globals()["MT_CHECKPOINT"] = str(args.mt)
    if args.links:
        os.environ["SCENE_LINKS"] = str(args.links)
        globals()["LINKS_CHECKPOINT"] = str(args.links)
    videos = sorted(getattr(split_videos(args.data), args.split))
    for excluded in args.exclude_seed or []:
        used = set(random.Random(excluded).sample(videos, min(args.videos, len(videos))))
        videos = [v for v in videos if v not in used]
    if args.still is None:
        chosen = random.Random(args.seed).sample(videos, min(args.videos, len(videos)))
    else:
        # Clips in the seed's order, kept while their camera is still enough.
        order = random.Random(args.seed).sample(videos, len(videos))
        chosen = []
        for video in order:
            if len(chosen) == args.videos:
                break
            if camera_shift(video) < args.still:
                chosen.append(video)
    report = {"split": args.split, "seed": args.seed, "videos": [v.name for v in chosen]}
    names = args.conditions.split(",") if args.conditions else list(CONDITIONS)
    for name in names:
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
