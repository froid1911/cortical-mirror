"""Benchmark one engine on a stratified sample of held-out videos."""

import argparse
import json
from pathlib import Path

from cortical_mirror.cli import create_engine
from cortical_mirror.data import split_videos
from cortical_mirror.evaluate import evaluate_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dmvfn-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--adaptation", type=float, default=0.3)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--videos", type=int, default=32)
    parser.add_argument("--frames", type=int, default=16)
    args = parser.parse_args()

    _, validation = split_videos(args.data)
    step = max(1, len(validation) // args.videos)
    videos = validation[::step][: args.videos]
    engine = create_engine(args)
    results = []
    for index, video in enumerate(videos, start=1):
        result = evaluate_video(engine, video, output=None, max_frames=args.frames, samples=0)
        result["video"] = str(video)
        results.append(result)
        print(
            f"{index:02d}/{len(videos)} {video.name}: "
            f"mae={result['model']['mae']:.5f} copy={result['copy']['mae']:.5f}",
            flush=True,
        )

    frames = sum(result["model"]["frames"] for result in results)
    aggregate = {
        "videos": len(results),
        "frames": frames,
        "model_mae": sum(r["model"]["mae"] * r["model"]["frames"] for r in results) / frames,
        "copy_mae": sum(r["copy"]["mae"] * r["copy"]["frames"] for r in results) / frames,
        "model_ssim": sum(r["model"]["ssim"] * r["model"]["frames"] for r in results) / frames,
        "copy_ssim": sum(r["copy"]["ssim"] * r["copy"]["frames"] for r in results) / frames,
        "beat_copy": sum(r["beat_copy"] * r["model"]["frames"] for r in results) / frames,
        "inference_ms_mean": sum(r["inference_ms_mean"] * r["frames_read"] for r in results)
        / sum(r["frames_read"] for r in results),
    }
    report = {"engine": args.engine, "aggregate": aggregate, "per_video": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
