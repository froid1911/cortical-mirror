"""Protocol 23: object files learn sharp shapes. One condition and one part per run."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from manipulation_report import learned  # tools/ is on the path when run as a script

from cortical_mirror.cortex.grouping import AssociationField, mean_object_iou
from cortical_mirror.cortex.learned_scene import LearnedScene
from cortical_mirror.cortex.mt import MotionCortex
from cortical_mirror.cortex.slots import MotionSlots
from cortical_mirror.scene.world import SpriteWorld

MOVING_SEEDS, MOVING_FRAMES, SCORED = range(9200, 9240), 300, range(100, 300, 20)
MANIPULATION_SEEDS, MANIPULATION_FRAMES = range(2300, 2340), 240


def moving(cortex, field, seeds, files: bool, silenced: bool) -> dict:
    ious = []
    for seed in seeds:
        world = SpriteWorld(seed=seed)
        slots = MotionSlots(field=field, device=cortex.device)
        scene = LearnedScene(cortex, slots, files=files, silenced=silenced)
        for t in range(MOVING_FRAMES):
            world.step()
            labels = scene.observe(world.render())
            if t not in SCORED:
                continue
            truth = np.zeros((world.height, world.width), np.int64)
            for thing in world.truth():
                truth[thing["visible"]] = thing["id"] + 1
            found = np.zeros_like(truth)
            for k, mask in scene.masks(labels).items():
                found[mask] = k
            ious.append(mean_object_iou(truth, found))
        print(f"seed {seed}: pixel object IoU {np.mean(ious):.3f} so far", flush=True)
    return {"pixel_object_iou": float(np.mean(ious)), "frames": len(ious), "per_frame": ious}


def paired(files: dict, plain: dict) -> dict:
    """Both scenes on the objects both matched: the mean error of each, per edit."""

    def key(r):
        return r["seed"], r["object"], r["edit"]

    plain_by = {key(r): r for r in plain["records"]}
    out = {"with_files": files, "without_files": plain}
    for edit in ("delete", "move"):
        pairs = [
            (r, plain_by[key(r)])
            for r in files["records"]
            if r["edit"] == edit and key(r) in plain_by
        ]
        out[f"{edit}_paired"] = {
            "objects": len(pairs),
            "with_files_mse": float(np.mean([a["scene"] for a, _ in pairs])),
            "without_files_mse": float(np.mean([b["scene"] for _, b in pairs])),
            "with_files_over_bitmap": float(
                np.mean([a["scene"] for a, _ in pairs]) / np.mean([a["bitmap"] for a, _ in pairs])
            ),
            "files_better": float(np.mean([a["scene"] < b["scene"] for a, b in pairs])),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mt", type=Path, required=True, help="the v0.9 MT adult")
    parser.add_argument("--field", type=Path, required=True, help="protocol 21's links")
    parser.add_argument("--part", choices=("moving", "manipulation", "paired"), required=True)
    parser.add_argument("--files", action="store_true", help="with object files")
    parser.add_argument("--silenced", action="store_true", help="MT gives zero velocity")
    parser.add_argument("--seeds", type=int, nargs=2, help="first and last+1 (development)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cortex = MotionCortex.load(args.mt, device=args.device, fast=True)
    cortex.fine.smoothing = cortex.coarse.smoothing = 3
    state = torch.load(args.field, map_location="cpu", weights_only=False)
    field = AssociationField(types=state["links"].shape[0], device=args.device)
    field.links = state["links"].to(args.device)

    if args.part == "moving":
        seeds = range(*args.seeds) if args.seeds else MOVING_SEEDS
        result = moving(cortex, field, seeds, args.files, args.silenced)
    elif args.part == "manipulation":
        seeds = range(*args.seeds) if args.seeds else MANIPULATION_SEEDS
        result = learned(cortex, field, seeds, MANIPULATION_FRAMES, files=args.files)
    else:
        seeds = range(*args.seeds) if args.seeds else MANIPULATION_SEEDS
        result = paired(
            learned(cortex, field, seeds, MANIPULATION_FRAMES, files=True),
            learned(cortex, field, seeds, MANIPULATION_FRAMES, files=False),
        )
    report = {
        "part": args.part,
        "files": args.files,
        "silenced": args.silenced,
        "seeds": [seeds.start, seeds.stop],
        "result": result,
    }
    shown = {k: v for k, v in result.items() if k not in ("per_frame", "records")}
    for k in ("with_files", "without_files"):
        if k in shown:
            shown[k] = {a: b for a, b in shown[k].items() if a != "records"}
    print(json.dumps(shown, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
