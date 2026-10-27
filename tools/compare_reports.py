"""Paired bootstrap comparison of two benchmark reports on the same videos.

Reports the difference in error reduction relative to persistence, in percentage
points, with a 95% interval from resampling videos, and on how many videos the
first engine had lower error than the second.
"""

import argparse
import json
import random
from pathlib import Path


def load(path: Path) -> dict[str, tuple[float, float, int]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        v["video"]: (v["model"]["mae"], v["copy"]["mae"], v["model"]["frames"])
        for v in report["per_video"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    first, second = load(args.first), load(args.second)
    videos = sorted(set(first) & set(second))
    if not videos:
        raise SystemExit("the reports share no videos")

    def gain(sample: list[str]) -> float:
        frames = sum(first[v][2] for v in sample)
        copy = sum(first[v][1] * first[v][2] for v in sample) / frames
        a = sum(first[v][0] * first[v][2] for v in sample) / frames
        b = sum(second[v][0] * second[v][2] for v in sample) / frames
        return 100.0 * (b - a) / copy

    rng = random.Random(args.seed)
    draws = sorted(gain([rng.choice(videos) for _ in videos]) for _ in range(args.samples))
    result = {
        "first": args.first.name,
        "second": args.second.name,
        "videos": len(videos),
        "difference_points": round(gain(videos), 2),
        "interval_95": [
            round(draws[int(0.025 * args.samples)], 2),
            round(draws[int(0.975 * args.samples) - 1], 2),
        ],
        "first_better_on_videos": sum(first[v][0] < second[v][0] for v in videos),
        "bootstrap_samples": args.samples,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
