"""Run the scene-level protocols (docs/protocols/08 onwards) and write a report."""

import argparse
import json
import time
from pathlib import Path

from cortical_mirror.scene.front import CortexFront, LuminanceFront, NormalisedFront
from cortical_mirror.scene.graph import NEWBORN_PERSISTENCE, SceneGraph
from cortical_mirror.scene.phenomena import (
    attention_trials,
    lighting,
    manipulation,
    object_discovery,
    object_permanence,
    parts_and_wholes,
    self_and_world,
    train_attention,
    train_motor,
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


def self_world() -> dict:
    """Protocol 12: learn a motor map, then test newborn and adult."""

    import numpy as np
    import torch

    motor = train_motor()
    ADULT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"motor": torch.tensor(motor)}, ADULT.with_name("scene-adult-eye.pt"))
    return {
        "adult_motor": motor.round(2).tolist(),
        "newborn": self_and_world(np.zeros_like(motor)),
        "adult": self_and_world(motor),
    }


def attention() -> dict:
    """Protocol 11: train the eye's gaze value, then test all three ways of looking."""

    import torch

    weights = train_attention()
    ADULT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weights": torch.tensor(weights)}, ADULT.with_name("scene-adult-gaze.pt"))
    return {
        "adult_weights": weights.round(4).tolist(),
        "newborn": attention_trials("random"),
        "surprise_reflex": attention_trials("surprise"),
        "adult": attention_trials("learned", weights),
    }


V1_ADULT = Path.home() / ".cache" / "cortical-mirror" / "cortex-adult-p1.pt"


def parts() -> dict:
    """Protocol 13."""

    return parts_and_wholes()


def light() -> dict:
    """Protocol 14: the v0.5 code, v0.4's brightness, and V1's code (reported)."""

    return lighting(
        {
            "normalised": NormalisedFront(),
            "luminance": LuminanceFront(),
            "v1": CortexFront(V1_ADULT),
        }
    )


def permanence_v05() -> dict:
    """Protocol 9 re-run for v0.5 on fresh test seeds, with a freshly raised adult."""

    adult, course = train_persistence()
    magic, _ = train_persistence(magic=True)
    seeds = range(3020, 3040)
    return {
        "adult_persistence": adult,
        "magic_persistence": magic,
        "adult_course": course,
        "newborn": object_permanence(NEWBORN_PERSISTENCE, seeds=seeds),
        "adult": object_permanence(adult, seeds=seeds),
        "built_in": object_permanence(60.0, seeds=seeds),
    }


def attention_v05() -> dict:
    """Protocol 11 re-run for v0.5 (gaze with a progress memory) on fresh test seeds."""

    weights = train_attention()
    seeds = range(5020, 5040)
    return {
        "adult_weights": weights.round(4).tolist(),
        "newborn": attention_trials("random", seeds=seeds),
        "surprise_reflex": attention_trials("surprise", seeds=seeds),
        "adult": attention_trials("learned", weights, seeds=seeds),
    }


def permanence_v06() -> dict:
    """Protocol 9 for v0.6 on fresh seeds, with a freshly raised adult."""

    adult, course = train_persistence()
    magic, _ = train_persistence(magic=True)
    seeds = range(3040, 3060)
    return {
        "adult_persistence": adult,
        "magic_persistence": magic,
        "adult_course": course,
        "newborn": object_permanence(NEWBORN_PERSISTENCE, seeds=seeds),
        "adult": object_permanence(adult, seeds=seeds),
        "built_in": object_permanence(60.0, seeds=seeds),
    }


def parts_v06() -> dict:
    """Protocol 13 for v0.6 on fresh seeds."""

    return parts_and_wholes(seeds=range(6020, 6040))


PROTOCOLS = {
    "discovery": object_discovery,
    "manipulation": manipulation,
    "permanence": permanence,
    "self_world": self_world,
    "attention": attention,
    "parts": parts,
    "lighting": light,
    "permanence_v05": permanence_v05,
    "attention_v05": attention_v05,
    "permanence_v06": permanence_v06,
    "parts_v06": parts_v06,
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
