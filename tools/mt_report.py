"""Protocol 16: motion cells, untrained vs adult, on the pre-registered stimuli."""

import argparse
import json
from pathlib import Path

from cortical_mirror.cortex.mt import MotionCortex
from cortical_mirror.cortex.phenomena import motion_cells


def summary(report: dict) -> dict:
    return {k: v for k, v in report.items() if not isinstance(v, list)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, required=True, help="the phase 1 cortex adult")
    parser.add_argument("--adult", type=Path, required=True, help="the MT adult")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "untrained": motion_cells(MotionCortex(v1=args.v1)),
        "adult": motion_cells(MotionCortex.load(args.adult)),
    }
    for name, result in report.items():
        print(name, json.dumps(summary(result)), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
