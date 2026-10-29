"""Protocol 20: objects from learned motion, on the pre-registered stimuli."""

import argparse
import json
from collections import deque
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
from cortical_mirror.cortex.slots import MotionSlots
from cortical_mirror.scene.world import SpriteWorld

TEST_SEEDS = range(9000, 9020)
MOVING_FRAMES, MOVING_SCORED = 300, range(100, 300, 20)
STILL_FROM, STILL_FRAMES, STILL_SCORED = 150, 180, (149, 179)


def view(world: SpriteWorld, device) -> torch.Tensor:
    """The world sent to the cortex at 128 × 96, ``[1, 1, 96, 128]`` in [0, 1]."""

    return F.avg_pool2d(torch.from_numpy(world.render())[None, None], 2).to(device)


def run(cortex, seed, frames, scored, still_from, conditions, gap=1) -> dict:
    """One world, every condition's slots fed the same frames; scores per condition.
    MT integrates over ``gap`` frames: V1 now against V1 ``gap`` frames ago."""

    world = SpriteWorld(seed=seed)
    past = deque(maxlen=gap + 1)
    slots = {name: make() for name, make in conditions.items()}
    scores = {name: [] for name in conditions}
    for t in range(frames):
        if t < still_from:
            world.step()
        past.append(cortex.v1(view(world, cortex.device)))
        if len(past) < 2:
            continue
        velocity = cortex.between(past[0], past[-1])[0] / (len(past) - 1)
        activity = past[-1][0]
        for name, s in slots.items():
            seen = torch.zeros_like(velocity) if name == "silenced" else velocity
            labels = s.step(seen, activity).numpy()
            if t in scored:
                truth = truth_on_grid(world.truth(), block=4)
                scores[name].append(
                    {
                        "frame": t,
                        "iou": mean_object_iou(truth, labels),
                        "ari": adjusted_rand_index(truth, labels),
                        "slots": int(len(np.unique(labels)) - 1),
                    }
                )
    return scores


def mean(rows, key, frame=None):
    values = [r[key] for r in rows if frame is None or r["frame"] == frame]
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mt", type=Path, required=True, help="the v0.9 MT adult")
    parser.add_argument("--field", type=Path, required=True, help="protocol 19's learned field")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs=2, help="first and last+1 (development)")
    parser.add_argument("--gap", type=int, default=4, help="frames MT integrates over")
    parser.add_argument("--smoothing", type=int, default=3, help="MT's pooling")
    parser.add_argument("--set", nargs="*", default=[], help="slot settings, name=value")
    args = parser.parse_args()

    seeds = range(*args.seeds) if args.seeds else TEST_SEEDS
    settings = {k: json.loads(v) for k, v in (s.split("=") for s in args.set)}
    cortex = MotionCortex.load(args.mt, device=args.device)
    if args.smoothing:
        cortex.fine.smoothing = cortex.coarse.smoothing = args.smoothing
    state = torch.load(args.field, map_location="cpu", weights_only=False)
    field = AssociationField(types=state["links"].shape[0], device=args.device)
    field.links = state["links"].to(args.device)

    def slots(**extra):
        return lambda: MotionSlots(field=field, device=args.device, **{**settings, **extra})

    conditions = {"slots": slots(), "silenced": slots(), "without_links": slots(beta=0.0)}
    moving = {name: [] for name in conditions}
    still = {name: [] for name in conditions}
    for seed in seeds:
        for name, rows in run(
            cortex, seed, MOVING_FRAMES, MOVING_SCORED, MOVING_FRAMES, conditions, args.gap
        ).items():
            moving[name] += rows
        for name, rows in run(
            cortex, seed, STILL_FRAMES, STILL_SCORED, STILL_FROM, conditions, args.gap
        ).items():
            still[name] += rows
        print(f"seed {seed}: slots IoU {mean(moving['slots'], 'iou'):.3f} so far", flush=True)

    before, after = STILL_SCORED
    summary = {
        name: {
            "moving_iou": mean(moving[name], "iou"),
            "moving_ari": mean(moving[name], "ari"),
            "moving_slots": mean(moving[name], "slots"),
            "still_iou_before": mean(still[name], "iou", before),
            "still_iou_after": mean(still[name], "iou", after),
        }
        for name in conditions
    }
    s = summary["slots"]
    verdict = {
        "moving_iou_criterion": s["moving_iou"] >= 0.5,
        "silenced_criterion": summary["silenced"]["moving_iou"] <= 0.1,
        "still_ratio": s["still_iou_after"] / max(s["still_iou_before"], 1e-9),
        "still_criterion": s["still_iou_after"] >= 0.8 * s["still_iou_before"],
    }
    print(json.dumps({"summary": summary, "verdict": verdict}, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "seeds": [seeds.start, seeds.stop],
        "settings": settings,
        "gap": args.gap,
        "smoothing": args.smoothing or cortex.fine.smoothing,
        "summary": summary,
        "verdict": verdict,
        "moving": moving,
        "still": still,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
