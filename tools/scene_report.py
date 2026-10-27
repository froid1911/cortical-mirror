"""Run the scene-level protocols (docs/protocols/08 onwards) and write a report."""

import argparse
import json
import time
from pathlib import Path

from cortical_mirror.scene.phenomena import manipulation, object_discovery

PROTOCOLS = {"discovery": object_discovery, "manipulation": manipulation}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocols", default=",".join(PROTOCOLS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = {}
    for name in args.protocols.split(","):
        started = time.time()
        report[name] = PROTOCOLS[name]()
        report[name]["seconds"] = round(time.time() - started, 1)
        print(name, json.dumps(report[name]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
