"""Protocol 18: self and world in learned motion, adult vs newborn efference map."""

import argparse
import json
from pathlib import Path

import torch

from cortical_mirror.cortex.efference import Efference
from cortical_mirror.cortex.mt import MotionCortex
from cortical_mirror.scene.phenomena import self_world_learned, train_efference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mt", type=Path, required=True, help="the v0.7 MT adult")
    parser.add_argument("--train-seeds", default="800:840")
    parser.add_argument("--test-seeds", default="8000:8020")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-map", type=Path)
    args = parser.parse_args()

    def seeds(text):
        start, stop = (int(v) for v in text.split(":"))
        return range(start, stop)

    mt = MotionCortex.load(args.mt)
    adult = train_efference(mt, seeds=seeds(args.train_seeds))
    if args.save_map:
        torch.save(adult.state_dict(), args.save_map)
    newborn = Efference(commands=9, types=mt.fine.types)
    report = {
        "adult": self_world_learned(mt, adult, seeds=seeds(args.test_seeds)),
        "newborn": self_world_learned(mt, newborn, seeds=seeds(args.test_seeds)),
        "adult_map": adult.map.tolist(),
    }
    for name in ("adult", "newborn"):
        print(name, json.dumps(report[name]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
