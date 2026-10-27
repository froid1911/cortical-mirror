"""Measure the scene level by the protocols in ``docs/protocols/`` (8 onwards).

The sprite world's truth is used here only, to score what the graph inferred;
the graph itself never sees it.
"""

from __future__ import annotations

import cv2
import numpy as np

from .graph import SceneGraph
from .world import SpriteWorld


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return int((a & b).sum()) / union if union else 0.0


def match(truth: list[np.ndarray], nodes: list[np.ndarray]) -> list[tuple[int, int, float]]:
    """One-to-one pairs (truth index, node index, IoU), greedily best first."""

    scores = sorted(
        ((iou(t, n), i, j) for i, t in enumerate(truth) for j, n in enumerate(nodes)),
        reverse=True,
    )
    used_truth, used_nodes, pairs = set(), set(), []
    for score, i, j in scores:
        if score <= 0 or i in used_truth or j in used_nodes:
            continue
        used_truth.add(i)
        used_nodes.add(j)
        pairs.append((i, j, score))
    return sorted(pairs)


def mean_iou(truth: list[np.ndarray], nodes: list[np.ndarray]) -> float:
    """Mean over objects of the matched IoU; an object without a node scores 0."""

    if not truth:
        return 1.0 if not nodes else 0.0
    return sum(score for _, _, score in match(truth, nodes)) / len(truth)


def motion_masks(frame: np.ndarray, earlier: np.ndarray) -> list[np.ndarray]:
    """The model-free baseline: connected components of what changed."""

    changed = (np.abs(frame - earlier) > 0.04).astype(np.uint8)
    changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(changed)
    return [labels == k for k in range(1, count) if stats[k, cv2.CC_STAT_AREA] >= 80]


def node_masks(graph: SceneGraph) -> list[np.ndarray]:
    return [graph._alpha(node) > 0.5 for node in graph.nodes]


def object_discovery(
    seeds=range(1000, 1040), frames: int = 300, sample_from: int = 200, every: int = 10
) -> dict:
    """Protocol 8."""

    totals = {name: {"iou": [], "count": []} for name in ("learning", "frozen", "motion_baseline")}
    samples = 0
    for seed in seeds:
        world = SpriteWorld(seed=seed, objects=1 + seed % 4)
        learning = SceneGraph(world.width, world.height, learn=True)
        frozen = SceneGraph(world.width, world.height, learn=False)
        recent: list[np.ndarray] = []
        for t in range(frames):
            world.step()
            image = world.render()
            learning.observe(image)
            frozen.observe(image)
            recent = [*recent, image][-9:]
            if t < sample_from or (t - sample_from) % every:
                continue
            samples += seed == seeds[0]
            truth = [o["visible"] for o in world.truth()]
            for name, masks in (
                ("learning", node_masks(learning)),
                ("frozen", node_masks(frozen)),
                ("motion_baseline", motion_masks(image, recent[0])),
            ):
                totals[name]["iou"].append(mean_iou(truth, masks))
                totals[name]["count"].append(float(len(masks) == len(truth)))
    report = {
        name: {
            "mean_iou": float(np.mean(values["iou"])),
            "count_accuracy": float(np.mean(values["count"])),
        }
        for name, values in totals.items()
    }
    report["samples"] = samples
    report["scenes"] = len(list(seeds))
    return report


def bitmap_edit(
    frame: np.ndarray, mask: np.ndarray, delete: bool = False, move: tuple | None = None
) -> np.ndarray:
    """What an editor without an explanation can do: cut the region out, fill the
    hole from its surroundings, and paste the cut pixels elsewhere."""

    pixels = np.rint(frame * 255).astype(np.uint8)
    hole = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    out = cv2.inpaint(pixels, hole, 5, cv2.INPAINT_TELEA).astype(np.float32) / 255.0
    if move:
        dx, dy = move
        h, w = mask.shape
        ys, xs = np.nonzero(mask)
        keep = (0 <= ys + dy) & (ys + dy < h) & (0 <= xs + dx) & (xs + dx < w)
        out[ys[keep] + dy, xs[keep] + dx] = frame[ys[keep], xs[keep]]
    return out


def move_offset(world: SpriteWorld, thing) -> tuple[int, int]:
    return (-30, 0) if thing.x > world.width / 2 else (30, 0)


def manipulation(seeds=range(2000, 2040), frames: int = 240) -> dict:
    """Protocol 10."""

    scores = {edit: {"graph": [], "bitmap": [], "unchanged": []} for edit in ("delete", "move")}
    objects = matched = 0
    for seed in seeds:
        world = SpriteWorld(seed=seed, objects=2 + seed % 2)
        graph = SceneGraph(world.width, world.height)
        for _ in range(frames):
            world.step()
            graph.observe(world.render())
        frame = world.render()
        truth = world.truth()
        masks = node_masks(graph)
        objects += len(truth)
        for t, n, score in match([o["visible"] for o in truth], masks):
            if score < 0.3:
                continue
            matched += 1
            thing = next(o for o in world.objects if o.id == truth[t]["id"])
            node = graph.nodes[n]
            offset = move_offset(world, thing)
            before = truth[t]["full"]
            after = np.roll(before, offset[::-1], axis=(0, 1))
            for edit, true_image, graph_image, bitmap_image in (
                (
                    "delete",
                    world.render(deleted={thing.id}),
                    graph.edit(delete={node.id}),
                    bitmap_edit(frame, masks[n], delete=True),
                ),
                (
                    "move",
                    world.render(moved={thing.id: offset}),
                    graph.edit(move={node.id: offset}),
                    bitmap_edit(frame, masks[n], move=offset),
                ),
            ):
                region = before | (after if edit == "move" else before)
                region = cv2.dilate(region.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                for name, image in (
                    ("graph", graph_image),
                    ("bitmap", bitmap_image),
                    ("unchanged", frame),
                ):
                    scores[edit][name].append(float(np.mean((image - true_image)[region] ** 2)))
    report = {"objects": objects, "matched": matched, "scenes": len(list(seeds))}
    for edit, values in scores.items():
        graph_mse = float(np.mean(values["graph"])) if values["graph"] else float("nan")
        bitmap_mse = float(np.mean(values["bitmap"])) if values["bitmap"] else float("nan")
        report[edit] = {
            "edits": len(values["graph"]),
            "graph_mse": graph_mse,
            "bitmap_mse": bitmap_mse,
            "unchanged_mse": float(np.mean(values["unchanged"])) if values["unchanged"] else None,
            "graph_over_bitmap": graph_mse / bitmap_mse if bitmap_mse else float("nan"),
            "graph_wins": float(
                np.mean([g < b for g, b in zip(values["graph"], values["bitmap"], strict=True)])
            )
            if values["graph"]
            else float("nan"),
        }
    return report
