"""Measure whether live learning lowers prediction error over one continuous scene.

UCF101 clips c01, c02, ... of one group are consecutive cuts of one source video,
so chaining them approximates time in front of one camera. Only test-split groups
are used. Scenes are ranked longest first, one per action class; `--offset`
skips the first ranked scenes, e.g. those used while developing the learner, so
the final settings can be scored on scenes that never influenced them.

`--video` scores any clip instead, repeated `--loop` times. Repetition means
later windows contain footage the learner has already seen; the report says so.

Learning steps run synchronously at `--steps-per-frame`, so results do not depend
on thread timing; the live server runs the same steps in a background thread.
Defaults match `cortical-mirror demo` on CUDA.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from cortical_mirror.data import split_videos
from cortical_mirror.engines import CorticalEngine, resolve_device
from cortical_mirror.online import OnlineLearner


def video_fps(video: Path) -> float:
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()
    return fps


def scene_frames(videos: list[Path], width: int, height: int, limit: int):
    count = 0
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        while count < limit:
            ok, frame = capture.read()
            if not ok:
                break
            yield cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            count += 1
        capture.release()


def run(engine: CorticalEngine, frames, steps_per_frame: float, bucket: int, fps: float):
    engine.reset()
    learner = engine.learner
    credit, pending, previous = 0.0, None, None
    curve, model_errors, copy_errors = [], [], []
    for index, frame in enumerate(frames):
        current = engine.predict(frame)
        if pending is not None:
            model_errors.append(float(np.mean(np.abs(pending.astype(np.float32) - frame)) / 255))
            copy_errors.append(float(np.mean(np.abs(previous.astype(np.float32) - frame)) / 255))
        if learner is not None and learner.ready():
            credit += steps_per_frame
            while credit >= 1.0:
                learner.step()
                credit -= 1.0
        if len(model_errors) == bucket:
            curve.append(
                {
                    "frame": index,
                    "seconds": round((index + 1) / fps, 2),
                    "model": float(np.mean(model_errors)),
                    "copy": float(np.mean(copy_errors)),
                    "steps": learner.stats().steps if learner else 0,
                    "rollbacks": learner.stats().rollbacks if learner else 0,
                }
            )
            model_errors, copy_errors = [], []
        pending, previous = current.image, frame
    return curve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="UCF101 root (scenes from the test split)")
    parser.add_argument("--video", type=Path, nargs="+", help="score these clips instead")
    parser.add_argument("--loop", type=int, default=1, help="repeat --video clips this often")
    parser.add_argument("--checkpoint", type=Path, help="omit to start from scratch")
    parser.add_argument("--scenes", type=int, default=6)
    parser.add_argument("--offset", type=int, default=0, help="skip the first ranked scenes")
    parser.add_argument("--frames", type=int, default=1800, help="maximum frames per scene")
    parser.add_argument("--bucket", type=int, default=150, help="frames per reported window")
    parser.add_argument("--steps-per-frame", type=float, default=0.5)
    parser.add_argument(
        "--learn-rate", type=float, help="default: 1e-3 from scratch, 2e-4 with --checkpoint"
    )
    parser.add_argument("--anchor", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--burn-in", type=int, default=8)
    parser.add_argument("--no-guard", action="store_true", help="disable rollbacks (ablation)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.data) == bool(args.video):
        parser.error("choose exactly one of --data and --video")
    device = resolve_device(args.device)
    learning_rate = args.learn_rate or (2e-4 if args.checkpoint else 1e-3)

    if args.video:
        scenes = [list(args.video) * args.loop]
    else:
        groups: dict[str, list[Path]] = {}
        for video in split_videos(args.data).test:
            groups.setdefault(video.stem.rsplit("_", 1)[0], []).append(video)
        # Longest scenes first, one per action class, for variety.
        ranked, seen = [], set()
        for name, videos in sorted(groups.items(), key=lambda item: -len(item[1])):
            action = name.split("_")[1]
            if action not in seen:
                ranked.append(sorted(videos))
                seen.add(action)
        scenes = ranked[args.offset : args.offset + args.scenes]

    config = {k: str(v) for k, v in vars(args).items()} | {"learning_rate": learning_rate}
    report = {"config": config, "scenes": []}
    for videos in scenes:
        fps = video_fps(videos[0])
        name = videos[0].stem if args.video else videos[0].stem.rsplit("_", 1)[0]
        result = {"scene": name, "clips": len(videos), "fps": fps, "loop": args.loop}
        for mode in ("frozen", "learning"):
            torch.manual_seed(0)
            if args.checkpoint:
                engine = CorticalEngine.from_checkpoint(args.checkpoint, device=device)
            else:
                engine = CorticalEngine.fresh(device=device)
            if mode == "learning":
                engine.learner = OnlineLearner(
                    engine.model,
                    learning_rate=learning_rate,
                    anchor_weight=args.anchor if args.checkpoint else 0.0,
                    steps_per_frame=args.steps_per_frame,
                    batch_size=args.batch_size,
                    burn_in=args.burn_in,
                    guard=not args.no_guard,
                )
                engine.learner.on_rollback = engine.reset
            frames = scene_frames(videos, engine.width, engine.height, args.frames)
            result[mode] = run(engine, frames, args.steps_per_frame, args.bucket, fps)
        frozen, learning = result["frozen"], result["learning"]
        print(
            f"{result['scene']}: "
            + " ".join(
                f"[{f['model'] / f['copy']:.3f}->{lr['model'] / lr['copy']:.3f}]"
                for f, lr in zip(frozen, learning, strict=False)
            ),
            flush=True,
        )
        report["scenes"].append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
