"""Measure the scene level by the protocols in ``docs/protocols/`` (8 onwards).

The sprite world's truth is used here only, to score what the graph inferred;
the graph itself never sees it.
"""

from __future__ import annotations

import math

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


def occlusion_world(seed: int) -> SpriteWorld:
    """Protocol 9's trial: one small object passing behind a wide screen."""

    world = SpriteWorld(seed=seed, objects=1, occluders=1, occluder_width=44, radius=(7, 9))
    thing = world.objects[0]
    thing.x, thing.y, thing.vx, thing.vy = 40.0, 96.0, 1.5, 0.0
    return world


def training_world(seed: int) -> SpriteWorld:
    """Protocol 9's training: objects, a screen, and nothing else arranged."""

    return SpriteWorld(
        seed=seed,
        objects=1 + seed % 3,
        occluders=1,
        occluder_width=30 + seed % 21,
        radius=(7, 12),
    )


def vanish_hidden(world: SpriteWorld) -> None:
    """The magic world: whatever is completely hidden ceases to exist."""

    gone = {t["id"] for t in world.truth() if t["full"].any() and not t["visible"].any()}
    world.objects = [o for o in world.objects if o.id not in gone]


def train_persistence(
    seeds=range(100, 300), frames: int = 300, magic: bool = False, persistence: float = 1.0
) -> tuple[float, list[float]]:
    """Raise a graph's persistence on training worlds; returns it and its course."""

    graph = SceneGraph(persistence=persistence, learn_persistence=True)
    trajectory = []
    for seed in seeds:
        graph.reset()
        world = training_world(seed)
        for _ in range(frames):
            world.step()
            if magic:
                vanish_hidden(world)
            graph.observe(world.render())
        trajectory.append(graph.persistence)
    return graph.persistence, trajectory


def _occlusion_trial(seed: int, vanish: bool, persistence: float, frames: int) -> dict:
    """Watch one trial; per frame: the image, the graph's expectation of it, the true
    position (None once the object is gone), and the nodes' positions."""

    world = occlusion_world(seed)
    graph = SceneGraph(persistence=persistence)
    record = {"images": [], "expected": [], "truth": [], "visible": [], "nodes": [], "hidden": None}
    for t in range(frames):
        world.step()
        truth = world.truth()
        if record["hidden"] is None and t > 5 and truth and not truth[0]["visible"].any():
            record["hidden"] = t
            if vanish:
                world.objects.clear()
        record["expected"].append(graph.expect())
        image = world.render()
        graph.observe(image)
        record["images"].append(image)
        thing = world.objects[0] if world.objects else None
        record["truth"].append(None if thing is None else (thing.x, thing.y))
        record["visible"].append(bool(thing) and bool(world.truth()[0]["visible"].any()))
        record["nodes"].append([(n.x, n.y) for n in graph.nodes])
    screen = world.occluders[0]
    record["right"] = int(screen.x + screen.mask.shape[1] / 2)
    return record


def object_permanence(persistence: float, seeds=range(3000, 3020), frames: int = 120) -> dict:
    """Protocol 9, for one persistence."""

    kept, errors, surprise = [], [], {"vanish": [], "reappear": []}
    for seed in seeds:
        reappear = _occlusion_trial(seed, False, persistence, frames)
        vanish = _occlusion_trial(seed, True, persistence, frames)
        hidden = reappear["hidden"]
        back = next(t for t in range(hidden, frames) if reappear["visible"][t])
        x, y = reappear["truth"][back - 1]
        near = min(
            (math.hypot(nx - x, ny - y) for nx, ny in reappear["nodes"][back - 1]),
            default=math.inf,
        )
        kept.append(near <= 8)
        if near <= 8:
            errors.append(near)
        region = (slice(72, 120), slice(reappear["right"], reappear["right"] + 36))
        for name, record in (("reappear", reappear), ("vanish", vanish)):
            surprise[name].append(
                float(
                    np.mean(
                        [
                            np.mean(
                                (record["expected"][t][region] - record["images"][t][region]) ** 2
                            )
                            for t in range(back, min(back + 15, frames))
                        ]
                    )
                )
            )
    vanish_mean = float(np.mean(surprise["vanish"]))
    reappear_mean = float(np.mean(surprise["reappear"]))
    return {
        "persistence": persistence,
        "kept": float(np.mean(kept)),
        "median_position_error": float(np.median(errors)) if errors else None,
        "surprise_vanish": vanish_mean,
        "surprise_reappear": reappear_mean,
        "violation_ratio": vanish_mean / reappear_mean if reappear_mean else float("nan"),
        "trials": len(kept),
    }


EYE_WORLD = (384, 288)
EYE_VIEW = (256, 192)


def _eye_world(seed: int) -> SpriteWorld:
    return SpriteWorld(seed=seed, width=EYE_WORLD[0], height=EYE_WORLD[1], objects=2)


def _saccade(gaze, command: int):
    from ..cortex.eye import COMMANDS
    from .eye import SACCADE

    dx, dy = COMMANDS[command]
    return gaze[0] + dx * SACCADE, gaze[1] + dy * SACCADE


def _inside(gaze) -> bool:
    w, h = EYE_VIEW
    return w / 2 <= gaze[0] <= EYE_WORLD[0] - w / 2 and h / 2 <= gaze[1] <= EYE_WORLD[1] - h / 2


def train_motor(seeds=range(400, 440), frames: int = 600, every: int = 5) -> np.ndarray:
    """Protocol 12's training: random saccades that keep the view in the world."""

    import random

    from .eye import EyeGraph, look

    graph = EyeGraph(EYE_VIEW)
    for seed in seeds:
        rng = random.Random(seed)
        world, gaze = _eye_world(seed), (EYE_WORLD[0] / 2, EYE_WORLD[1] / 2)
        graph.reset_scene()
        for t in range(frames):
            command = 0
            if t >= 20 and t % every == every - 1:
                command = rng.choice([c for c in range(1, 9) if _inside(_saccade(gaze, c))])
                gaze = _saccade(gaze, command)
            world.step()
            graph.observe(look(world.render(), gaze, EYE_VIEW), command)
    return graph.motor.copy()


def self_and_world(motor: np.ndarray, seeds=range(4000, 4020), settle: int = 30) -> dict:
    """Protocol 12, for one motor map."""

    import copy
    import random

    from .eye import EyeGraph, look

    fractions = {"commanded": [], "uncommanded": []}
    from_memory, unchanged = [], []
    for seed in seeds:
        rng = random.Random(seed)
        world, gaze = _eye_world(seed), (EYE_WORLD[0] / 2, EYE_WORLD[1] / 2)
        graph = EyeGraph(EYE_VIEW, motor=motor, learn_motor=False)
        for _ in range(settle):
            world.step()
            graph.observe(look(world.render(), gaze, EYE_VIEW), 0)
        current = look(world.render(), gaze, EYE_VIEW)
        command = rng.choice([c for c in range(1, 9) if _inside(_saccade(gaze, c))])
        target = _saccade(gaze, command)
        length = math.hypot(target[0] - gaze[0], target[1] - gaze[1])
        predicted, sure = graph.predict_view(command)
        world.step()
        view = look(world.render(), target, EYE_VIEW)
        if sure.any():
            from_memory.append(float(np.mean((predicted[sure] - view[sure]) ** 2)))
            unchanged.append(float(np.mean((current[sure] - view[sure]) ** 2)))
        for name, told in (("commanded", command), ("uncommanded", 0)):
            subject = copy.deepcopy(graph)
            before = subject.world_shift.copy()
            subject.observe(view, told)
            moved = subject.world_shift - before
            fractions[name].append(float(np.hypot(*moved)) / length)
    memory, still = float(np.mean(from_memory)), float(np.mean(unchanged))
    return {
        "commanded": float(np.mean(fractions["commanded"])),
        "uncommanded": float(np.mean(fractions["uncommanded"])),
        "prediction_error_from_memory": memory,
        "prediction_error_unchanged": still,
        "prediction_ratio": memory / still if still else float("nan"),
        "trials": len(fractions["commanded"]),
    }


# Gaze positions that keep the view inside the world: the noise and the newcomer
# are placed at their corners, where a fixation can cover them fully.
CORNERS = [(128, 96), (256, 96), (128, 192), (256, 192)]
FOVEA = 96


class AttentionTrial:
    """Protocol 11's world: moving objects, a patch of TV noise, and a newcomer."""

    def __init__(self, seed: int, novel_at: int = 300):
        self.world = SpriteWorld(seed=seed, width=EYE_WORLD[0], height=EYE_WORLD[1], objects=4)
        self.newcomer = self.world.objects.pop()
        self.noise = CORNERS[seed % 4]
        self.novel_at = novel_at
        self.rng = np.random.default_rng(seed)

    def frame(self, t: int, gaze=(192, 144)) -> np.ndarray:
        if t == self.novel_at:
            # Where the gaze is not, and the noise is not.
            far = max(
                (c for c in CORNERS if c != self.noise),
                key=lambda c: math.hypot(c[0] - gaze[0], c[1] - gaze[1]),
            )
            self.newcomer.x, self.newcomer.y = float(far[0]), float(far[1])
            self.world.objects.append(self.newcomer)
        self.world.step()
        image = self.world.render()
        x, y = self.noise
        half = FOVEA // 2
        image[y - half : y + half, x - half : x + half] = self.rng.random((FOVEA, FOVEA))
        return image


def _fovea_overlap(gaze, centre) -> float:
    dx = max(0.0, FOVEA - abs(gaze[0] - centre[0]))
    dy = max(0.0, FOVEA - abs(gaze[1] - centre[1]))
    return dx * dy / FOVEA**2


def _attention_run(seed: int, attention, frames: int, novel_at: int):
    from ..cortex.eye import COMMANDS
    from .eye import SACCADE, EyeGraph, look

    motor = np.array(COMMANDS, float) * SACCADE
    graph = EyeGraph(EYE_VIEW, motor=motor, learn_motor=False, fovea=FOVEA)
    trial = AttentionTrial(seed, novel_at)
    gaze = (EYE_WORLD[0] / 2, EYE_WORLD[1] / 2)
    fixations, found = [], None
    attention.reset()
    for t in range(frames):
        command = 0
        if graph.inner.background is not None:
            command = attention.step(graph, t, allowed=lambda c, g=gaze: _inside(_saccade(g, c)))
        gaze = _saccade(gaze, command)
        if command:
            fixations.append(gaze)
        image = trial.frame(t, gaze)
        graph.observe(look(image, gaze, EYE_VIEW), command)
        newcomer = trial.newcomer
        if found is None and t >= novel_at and _fovea_overlap(gaze, (newcomer.x, newcomer.y)) > 0:
            if abs(gaze[0] - newcomer.x) < FOVEA / 2 and abs(gaze[1] - newcomer.y) < FOVEA / 2:
                found = t - novel_at
    return graph, trial, fixations, found


def attention_trials(
    policy: str, weights=None, seeds=range(5000, 5020), frames: int = 600, novel_at: int = 300
) -> dict:
    """Protocol 11, for one way of choosing where to look."""

    from .attention import Attention

    captured, errors, novelty = [], [], []
    for seed in seeds:
        attention = Attention(weights=weights, policy=policy, seed=seed)
        graph, trial, fixations, found = _attention_run(seed, attention, frames, novel_at)
        captured.extend(_fovea_overlap(g, trial.noise) >= 0.5 for g in fixations)
        # The graph's canvas, laid over the world (its gaze began at the world's centre).
        ox = graph.start[0] - EYE_WORLD[0] // 2
        oy = graph.start[1] - EYE_WORLD[1] // 2
        known = graph.background[oy : oy + EYE_WORLD[1], ox : ox + EYE_WORLD[0]]
        outside = np.ones(known.shape, bool)
        x, y = trial.noise
        outside[y - FOVEA // 2 : y + FOVEA // 2, x - FOVEA // 2 : x + FOVEA // 2] = False
        errors.append(float(np.mean((known - trial.world.background)[outside] ** 2)))
        novelty.append(frames - novel_at if found is None else found)
    return {
        "policy": policy,
        "noise_capture": float(np.mean(captured)) if captured else 0.0,
        "world_error": float(np.mean(errors)),
        "time_to_novelty": float(np.median(novelty)),
        "trials": len(errors),
    }


def train_attention(seeds=range(500, 540), frames: int = 600, explore: float = 0.2) -> np.ndarray:
    """Protocol 11's training: one eye learning across all training worlds."""

    from .attention import Attention

    attention = Attention(learn=True, explore=explore)
    for seed in seeds:
        attention.rng.seed(seed)
        _attention_run(seed, attention, frames, novel_at=frames // 2)
    return attention.weights.copy()
