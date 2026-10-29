"""Protocol 19: grouping learned from common fate, on the pre-registered stimuli."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from cortical_mirror.cortex.grouping import (
    AssociationField,
    adjusted_rand_index,
    mean_object_iou,
    truth_on_grid,
)
from cortical_mirror.cortex.mt import MotionCortex
from cortical_mirror.scene.world import SpriteWorld

TRAIN_SEEDS, TRAIN_FRAMES = range(900, 960), 300
TEST_SEEDS, TEST_FRAMES = range(9000, 9020), range(100, 300, 20)


def view(world: SpriteWorld, device) -> torch.Tensor:
    """The world sent to the cortex at 128 × 96, ``[1, 1, 96, 128]`` in [0, 1]."""

    return F.avg_pool2d(torch.from_numpy(world.render())[None, None], 2).to(device)


def train(cortex: MotionCortex, field: AssociationField, log) -> None:
    for seed in TRAIN_SEEDS:
        world, started = SpriteWorld(seed=seed), time.time()
        cortex.reset()
        for _ in range(TRAIN_FRAMES):
            world.step()
            velocity = cortex.step(view(world, cortex.device))
            if velocity is None:
                continue
            # MT's velocity, in retinal px per frame, pooled to V1's grid.
            field.learn(cortex.previous[0], F.avg_pool2d(velocity, cortex.stride)[0])
        log(f"trained seed {seed} in {time.time() - started:.1f}s", flush=True)


def test(cortex: MotionCortex, fields: dict) -> dict:
    scores = {name: {"iou": [], "ari": [], "groups": []} for name in fields}
    for seed in TEST_SEEDS:
        world = SpriteWorld(seed=seed)
        for frame in range(TEST_FRAMES.stop):
            world.step()
            if frame not in TEST_FRAMES:
                continue
            # Features of the current frame only: no motion at test.
            activity = cortex.v1(view(world, cortex.device))[0]
            truth = truth_on_grid(world.truth(), block=4)
            active = (activity.sum(0) > 1e-6).cpu().numpy()
            for name, field in fields.items():
                found = field.groups(activity).numpy()
                scores[name]["iou"].append(mean_object_iou(truth, found))
                scores[name]["ari"].append(adjusted_rand_index(truth[active], found[active]))
                scores[name]["groups"].append(int(found.max()))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mt", type=Path, required=True, help="the v0.9 MT adult")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-field", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cortex = MotionCortex.load(args.mt, device=args.device)
    types = cortex.hierarchy.configs[0].cells
    learned = AssociationField(types=types, device=args.device)
    train(cortex, learned, print)
    if args.save_field:
        args.save_field.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"links": learned.links.cpu(), "offsets": learned.offsets}, args.save_field)

    newborn = AssociationField(types=types, device=args.device)
    similarity = AssociationField(types=types, device=args.device)
    similarity.links[:] = torch.eye(types, device=args.device)[:, :, None]
    scores = test(cortex, {"learned": learned, "newborn": newborn, "similarity": similarity})

    summary = {
        name: {
            "mean_object_iou": float(np.nanmean(s["iou"])),
            "ari": float(np.mean(s["ari"])),
            "mean_groups": float(np.mean(s["groups"])),
            "frames": len(s["iou"]),
        }
        for name, s in scores.items()
    }
    learned_ari, baseline_ari = summary["learned"]["ari"], summary["similarity"]["ari"]
    verdict = {
        "ari_over_similarity": learned_ari - baseline_ari,
        "ari_criterion": learned_ari >= baseline_ari + 0.05,
        "iou_criterion": summary["learned"]["mean_object_iou"] >= 0.5,
    }
    print(json.dumps({"summary": summary, "verdict": verdict}, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"summary": summary, "verdict": verdict, "frames": scores}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
