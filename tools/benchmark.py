"""Benchmark one engine on videos stratified by action class from a reserved split.

The default `test` split (official UCF101 split-1 test groups) shares no source
video with training, and nothing in this repository was tuned on it. Use
`--split validation` while choosing checkpoints or routing settings.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from cortical_mirror.cli import add_engine_arguments, create_engine
from cortical_mirror.data import split_videos
from cortical_mirror.evaluate import Perceptual, evaluate_video


def stratified(videos: list[Path], per_class: int) -> list[Path]:
    by_class: dict[str, list[Path]] = defaultdict(list)
    for video in sorted(videos):
        by_class[video.parent.name].append(video)
    chosen = []
    for _, members in sorted(by_class.items()):
        step = max(1, len(members) // per_class)
        chosen.extend(members[::step][:per_class])
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_arguments(parser)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("test", "validation", "all"),
        default="test",
        help="`all` scores every video, e.g. a folder of fresh webcam recordings",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--lpips", action="store_true", help="also report LPIPS (pip lpips)")
    args = parser.parse_args()
    if args.learn or args.from_scratch:
        # The live learner runs on its own thread, so results would depend on timing.
        parser.error("measure online learning with tools/live_learning.py, not this benchmark")

    split = split_videos(args.data)
    if args.split == "all":
        videos = sorted(split.train + split.validation + split.test)
    else:
        videos = stratified(getattr(split, args.split), args.per_class)
    engine = create_engine(args)
    perceptual = Perceptual(args.device) if args.lpips else None
    results = []
    for index, video in enumerate(videos, start=1):
        result = evaluate_video(
            engine, video, output=None, max_frames=args.frames, samples=0, perceptual=perceptual
        )
        result["video"] = str(video)
        results.append(result)
        print(
            f"{index:03d}/{len(videos)} {video.name}: "
            f"mae={result['model']['mae']:.5f} copy={result['copy']['mae']:.5f}",
            flush=True,
        )

    frames = sum(result["model"]["frames"] for result in results)

    def weighted(side: str, key: str) -> float:
        return sum(r[side][key] * r[side]["frames"] for r in results) / frames

    aggregate = {
        "split": args.split,
        "videos": len(results),
        "frames": frames,
        "model_mae": weighted("model", "mae"),
        "copy_mae": weighted("copy", "mae"),
        "model_ssim": weighted("model", "ssim"),
        "copy_ssim": weighted("copy", "ssim"),
        "beat_copy": sum(r["beat_copy"] * r["model"]["frames"] for r in results) / frames,
        "inference_ms_mean": sum(r["inference_ms_mean"] * r["frames_read"] for r in results)
        / sum(r["frames_read"] for r in results),
    }
    if args.lpips:
        aggregate["model_lpips"] = weighted("model", "lpips")
        aggregate["copy_lpips"] = weighted("copy", "lpips")
    config = {key: str(value) for key, value in vars(args).items()}
    report = {"engine": args.engine, "config": config, "aggregate": aggregate, "per_video": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
