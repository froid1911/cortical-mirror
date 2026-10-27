"""Run the scene-level protocols (docs/protocols/08 onwards) and write a report."""

import argparse
import json
import time
from pathlib import Path

from cortical_mirror.scene.graph import NEWBORN_PERSISTENCE, SceneGraph
from cortical_mirror.scene.phenomena import (
    manipulation,
    object_discovery,
    object_permanence,
    train_persistence,
)

ADULT = Path("checkpoints/scene-adult.pt")


def permanence() -> dict:
    """Protocol 9: raise an adult and a magic-world control, then test all conditions."""

    adult, adult_course = train_persistence()
    magic, magic_course = train_persistence(magic=True)
    graph = SceneGraph(persistence=adult, learn_persistence=True)
    ADULT.parent.mkdir(parents=True, exist_ok=True)
    graph.save(ADULT)
    return {
        "adult_persistence": adult,
        "magic_persistence": magic,
        "adult_course": adult_course,
        "magic_course": magic_course,
        "newborn": object_permanence(NEWBORN_PERSISTENCE),
        "adult": object_permanence(adult),
        "built_in": object_permanence(60.0),
    }


PROTOCOLS = {
    "discovery": object_discovery,
    "manipulation": manipulation,
    "permanence": permanence,
}


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
