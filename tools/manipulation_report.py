"""Protocol 22: manipulating a scene of learned objects, beside protocol 10's graph."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from cortical_mirror.cortex.grouping import AssociationField
from cortical_mirror.cortex.learned_scene import LearnedScene
from cortical_mirror.cortex.mt import MotionCortex
from cortical_mirror.cortex.slots import MotionSlots
from cortical_mirror.scene.phenomena import bitmap_edit, manipulation, move_offset
from cortical_mirror.scene.world import SpriteWorld

SEEDS, FRAMES = range(2100, 2140), 240


def matches(truth: list[dict], masks: dict[int, np.ndarray]) -> list[tuple[dict, int, float]]:
    """Objects to slots, one-to-one, greedily by the share of the object a slot covers."""

    scores = sorted(
        (
            (float((o["visible"] & m).sum() / max(o["visible"].sum(), 1)), i, k)
            for i, o in enumerate(truth)
            for k, m in masks.items()
        ),
        reverse=True,
    )
    used_objects, used_slots, pairs = set(), set(), []
    for share, i, k in scores:
        if i in used_objects or k in used_slots or share <= 0:
            continue
        used_objects.add(i)
        used_slots.add(k)
        pairs.append((truth[i], k, share))
    return pairs


def learned(cortex, field, seeds, frames, files: bool = False) -> dict:
    scores = {e: {"scene": [], "bitmap": [], "unchanged": []} for e in ("delete", "move")}
    records = []
    objects = matched = 0
    for seed in seeds:
        world = SpriteWorld(seed=seed, objects=2 + seed % 2)
        slots = MotionSlots(field=field, device=cortex.device)
        scene = LearnedScene(cortex, slots, files=files)
        for _ in range(frames):
            world.step()
            labels = scene.observe(world.render())
        frame, truth, masks = world.render(), world.truth(), scene.masks(labels)
        objects += len(truth)
        for t, k, share in matches(truth, masks):
            if share < 0.5:
                continue
            matched += 1
            thing = next(o for o in world.objects if o.id == t["id"])
            offset = move_offset(world, thing)
            before = t["full"]
            after = np.roll(before, offset[::-1], axis=(0, 1))
            for edit, true_image, scene_image, bitmap_image in (
                (
                    "delete",
                    world.render(deleted={thing.id}),
                    scene.edit(frame, labels, delete={k}),
                    bitmap_edit(frame, masks[k], delete=True),
                ),
                (
                    "move",
                    world.render(moved={thing.id: offset}),
                    scene.edit(frame, labels, move={k: offset}),
                    bitmap_edit(frame, masks[k], move=offset),
                ),
            ):
                region = before | (after if edit == "move" else before)
                region = cv2.dilate(region.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                errors = {}
                for name, image in (
                    ("scene", scene_image),
                    ("bitmap", bitmap_image),
                    ("unchanged", frame),
                ):
                    errors[name] = float(np.mean((image - true_image)[region] ** 2))
                    scores[edit][name].append(errors[name])
                records.append({"seed": seed, "object": int(thing.id), "edit": edit, **errors})
        print(f"seed {seed}: {matched} of {objects} objects matched so far", flush=True)
    report = {
        "objects": objects,
        "matched": matched,
        "matched_share": matched / max(objects, 1),
        "records": records,
    }
    for edit, values in scores.items():
        scene_mse, bitmap_mse = float(np.mean(values["scene"])), float(np.mean(values["bitmap"]))
        report[edit] = {
            "edits": len(values["scene"]),
            "scene_mse": scene_mse,
            "bitmap_mse": bitmap_mse,
            "unchanged_mse": float(np.mean(values["unchanged"])),
            "scene_over_bitmap": scene_mse / bitmap_mse,
            "scene_wins": float(
                np.mean([s < b for s, b in zip(values["scene"], values["bitmap"], strict=True)])
            ),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mt", type=Path, required=True, help="the v0.9 MT adult")
    parser.add_argument("--field", type=Path, required=True, help="protocol 21's links")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs=2, help="first and last+1 (development)")
    parser.add_argument("--no-yardstick", action="store_true", help="skip protocol 10's graph")
    args = parser.parse_args()

    seeds = range(*args.seeds) if args.seeds else SEEDS
    cortex = MotionCortex.load(args.mt, device=args.device, fast=True)
    cortex.fine.smoothing = cortex.coarse.smoothing = 3
    state = torch.load(args.field, map_location="cpu", weights_only=False)
    field = AssociationField(types=state["links"].shape[0], device=args.device)
    field.links = state["links"].to(args.device)

    report = {"seeds": [seeds.start, seeds.stop], "learned": learned(cortex, field, seeds, FRAMES)}
    if not args.no_yardstick:
        report["engineered_graph"] = manipulation(seeds=seeds, frames=FRAMES)
    s = report["learned"]
    report["verdict"] = {
        "delete_criterion": s["delete"]["scene_over_bitmap"] <= 0.75,
        "move_criterion": s["move"]["scene_over_bitmap"] < 1.0,
        "matched_criterion": s["matched_share"] >= 0.8,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "seeds"}, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
