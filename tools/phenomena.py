"""Run the cortex phenomenon protocols (docs/protocols/) and write a report.

The network under test is the untrained network (--untrained), a saved one
(--checkpoint), or either of them after raising it on a video (--experience):
the newborn condition. Raising learns online, frame by frame, exactly as the
live mirror does.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import torch

from cortical_mirror.cortex.engine import CortexEngine
from cortical_mirror.cortex.phenomena import oddball_adaptation, orientation_tuning

PROTOCOLS = {
    "orientation": orientation_tuning,
    "oddball": oddball_adaptation,
}


def raise_on(engine: CortexEngine, video: Path, frames: int) -> dict:
    """Let the engine learn from `frames` consecutive frames, looping the video."""

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"cannot read video {video}")
    seen, loops, code_errors, persistence = 0, 0, [], []
    seen_at_loop_start = 0
    while seen < frames:
        ok, frame = capture.read()
        if not ok:
            if seen == seen_at_loop_start:
                raise SystemExit(f"cannot read any frame from {video}")
            loops += 1
            seen_at_loop_start = seen
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        engine.predict(frame)
        seen += 1
        if seen > frames - 300:
            code_errors.append(engine.last_code_error)
            persistence.append(engine.last_persistence_error)
    capture.release()
    return {
        "video": video.name,
        "frames": frames,
        "loops": loops,
        "last_300_code_error_over_persistence": sum(code_errors) / max(sum(persistence), 1e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--untrained", action="store_true")
    source.add_argument("--checkpoint", type=Path)
    parser.add_argument("--phase", type=int, help="default 1; must match a --checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experience", type=Path, help="raise the network on this video first")
    parser.add_argument("--experience-frames", type=int, default=3600)
    parser.add_argument("--protocols", default=",".join(PROTOCOLS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.checkpoint:
        engine = CortexEngine.load(args.checkpoint, device=args.device, learn=True)
        if args.phase is not None and args.phase != engine.phase:
            parser.error(f"--phase {args.phase} contradicts the checkpoint's phase {engine.phase}")
    else:
        engine = CortexEngine(device=args.device, learn=True, phase=args.phase or 1, seed=args.seed)
    report = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "age_frames_before": engine.age_frames,
    }
    if args.experience:
        started = time.time()
        report["experience"] = raise_on(engine, args.experience, args.experience_frames)
        report["experience"]["seconds"] = round(time.time() - started, 1)
    report["age_frames"] = engine.age_frames
    for name in args.protocols.split(","):
        started = time.time()
        report[name] = PROTOCOLS[name](engine.hierarchy)
        report[name]["seconds"] = round(time.time() - started, 1)
        summary = {k: v for k, v in report[name].items() if not isinstance(v, list)}
        print(name, json.dumps(summary), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
