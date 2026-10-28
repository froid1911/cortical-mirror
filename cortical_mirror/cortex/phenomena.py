"""Measure cortex-like phenomena the way neuroscience measured them.

Every measure follows a protocol in ``docs/protocols/``. Measurements never change
the network they measure: tuning is read from a frozen settle, and sequence
experiments run on a copy.
"""

from __future__ import annotations

import copy
import math
import random

import torch
from torch.nn import functional as F

from .hierarchy import Hierarchy
from .retina import Retina

WIDTH, HEIGHT = 128, 96


def grating(
    orientation_deg: float,
    phase: float = 0.0,
    period: float = 8.0,
    contrast: float = 0.8,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> torch.Tensor:
    """A static sinusoidal grating, luminance ``[1, 1, H, W]`` in [0, 1]."""

    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    angle = math.radians(orientation_deg)
    wave = torch.cos(2 * math.pi * (xx * math.cos(angle) + yy * math.sin(angle)) / period + phase)
    return (0.5 + 0.5 * contrast * wave).float()[None, None]


def _encode_static(network: Hierarchy, frames: torch.Tensor) -> torch.Tensor:
    retina = Retina(WIDTH, HEIGHT, lagged=network.in_channels == 4, device=network.device)
    code = retina.encode(frames)
    # A steady stimulus: lagged channels see the same image as the current ones.
    return retina.encode(frames) if network.in_channels == 4 else code


@torch.no_grad()
def orientation_tuning(
    network: Hierarchy, period: float = 8.0, orientations: int = 12, phases: int = 8
) -> dict:
    """Protocol 1. Returns per-cell OSI, the median, and the fraction with OSI > 0.3."""

    stimuli = torch.cat(
        [
            grating(180.0 * o / orientations, 2 * math.pi * p / phases, period)
            for o in range(orientations)
            for p in range(phases)
        ]
    )
    code = _encode_static(network, stimuli)
    activity = network.settle(code)[0]
    centre = activity[:, :, HEIGHT // 4 - 4 : HEIGHT // 4 + 4, WIDTH // 4 - 4 : WIDTH // 4 + 4]
    responses = centre.mean((2, 3)).view(orientations, phases, -1).mean(1).cpu()
    angles = torch.arange(orientations) * math.pi / orientations
    vector = (responses * torch.exp(2j * angles)[:, None]).sum(0).abs()
    total = responses.sum(0)
    responsive = total > 1e-4
    osi = (vector[responsive] / total[responsive]).tolist()
    return {
        "osi": osi,
        "median_osi": float(torch.tensor(osi).median()) if osi else 0.0,
        "fraction_selective": sum(value > 0.3 for value in osi) / max(len(osi), 1),
        "responsive_cells": int(responsive.sum()),
        "curves": responses.T.tolist(),
    }


def mismatch_indices(deviant: float, standard: float, control: float) -> dict:
    """Parras et al. (2017): repetition suppression, prediction error, mismatch."""

    norm = math.sqrt(deviant**2 + standard**2 + control**2) or 1.0
    return {
        "iRS": (control - standard) / norm,
        "iPE": (deviant - control) / norm,
        "iMM": (deviant - standard) / norm,
    }


def _oddball_sequence(trials: int, p_deviant: float, rng: random.Random) -> list[bool]:
    sequence = [False] * 10
    while len(sequence) < trials:
        deviant = not sequence[-1] and rng.random() < p_deviant
        sequence.append(deviant)
    return sequence


@torch.no_grad()
def _onset_errors(
    network: Hierarchy, orientations: list[float], frames_per_stimulus: int
) -> list[float]:
    network = copy.deepcopy(network)
    network.reset_state()
    retina = Retina(WIDTH, HEIGHT, lagged=network.in_channels == 4, device=network.device)
    stimuli = {angle: grating(angle) for angle in set(orientations)}
    errors = []
    for angle in orientations:
        for frame in range(frames_per_stimulus):
            result = network.step(retina.encode(stimuli[angle]), learn=True)
            if frame == 0:
                errors.append(result.code_error)
    return errors


def oddball_adaptation(
    network: Hierarchy,
    standard_deg: float = 0.0,
    deviant_deg: float = 90.0,
    trials: int = 200,
    p_deviant: float = 0.1,
    frames_per_stimulus: int = 4,
    seed: int = 0,
    skip: int = 20,
) -> dict:
    """Protocol 2. Onset prediction errors for deviant, standard and control."""

    rng = random.Random(seed)
    oddball = _oddball_sequence(trials, p_deviant, rng)
    oddball_angles = [deviant_deg if is_deviant else standard_deg for is_deviant in oddball]
    control_angles = [rng.choice([18.0 * k for k in range(10)]) for _ in range(trials)]
    oddball_errors = _onset_errors(network, oddball_angles, frames_per_stimulus)
    control_errors = _onset_errors(network, control_angles, frames_per_stimulus)

    def mean(values):
        values = [v for v in values if math.isfinite(v)]
        return sum(values) / len(values) if values else float("nan")

    dev = mean(
        [
            e
            for i, (a, e) in enumerate(zip(oddball_angles, oddball_errors, strict=True))
            if i >= skip and a == deviant_deg
        ]
    )
    std = mean(
        [
            e
            for i, (a, e) in enumerate(zip(oddball_angles, oddball_errors, strict=True))
            if i >= skip and a == standard_deg
        ]
    )
    ctr = mean(
        [
            e
            for i, (a, e) in enumerate(zip(control_angles, control_errors, strict=True))
            if i >= skip and a == deviant_deg
        ]
    )
    return {"DEV": dev, "STD": std, "CTR": ctr, **mismatch_indices(dev, std, ctr)}


def direction_index(preferred: float, opposite: float) -> float:
    total = preferred + opposite
    return (preferred - opposite) / total if total > 0 else 0.0


@torch.no_grad()
def direction_selectivity(
    network: Hierarchy,
    period: float = 8.0,
    speed: float = 1.0,
    directions: int = 12,
    frames: int = 16,
) -> dict:
    """Protocol 3. DSI per cell from drifting gratings seen through the lagged retina."""

    codes = []
    for d in range(directions):
        angle = 360.0 * d / directions
        retina = Retina(WIDTH, HEIGHT, lagged=network.in_channels == 4, device=network.device)
        for t in range(frames):
            code = retina.encode(grating(angle, -2 * math.pi * speed * t / period, period))
            if t >= 2:
                codes.append(code)
    code = torch.cat(codes)
    activity = network.settle(code)[0]
    centre = activity[:, :, HEIGHT // 4 - 4 : HEIGHT // 4 + 4, WIDTH // 4 - 4 : WIDTH // 4 + 4]
    responses = centre.mean((2, 3)).view(directions, frames - 2, -1).mean(1).cpu()
    dsi = []
    for cell in range(responses.shape[1]):
        curve = responses[:, cell]
        best = int(curve.argmax())
        if float(curve[best]) <= 1e-4:
            continue
        dsi.append(
            direction_index(float(curve[best]), float(curve[(best + directions // 2) % directions]))
        )
    return {
        "dsi": dsi,
        "median_dsi": float(torch.tensor(dsi).median()) if dsi else 0.0,
        "fraction_selective": sum(value > 0.3 for value in dsi) / max(len(dsi), 1),
        "responsive_cells": len(dsi),
        "curves": responses.T.tolist(),
    }


def suppression_index(curve: list[float]) -> float:
    """(peak − response to the largest stimulus) / peak; 0 when nothing responds."""

    peak = max(curve)
    return (peak - curve[-1]) / peak if peak > 0 else 0.0


def _preferred_orientations(network: Hierarchy, orientations: int = 12) -> list[float]:
    curves = orientation_tuning(network, orientations=orientations)["curves"]
    return [
        180.0 * max(range(orientations), key=curve.__getitem__) / orientations for curve in curves
    ]


@torch.no_grad()
def _centre_responses(network: Hierarchy, frames: torch.Tensor, feedback: bool, chunk: int = 64):
    """Each cell's activity at the central 2 x 2 V1 positions, per stimulus."""

    out = []
    for begin in range(0, len(frames), chunk):
        code = _encode_static(network, frames[begin : begin + chunk])
        activity = network.settle(code, feedback=feedback)[0]
        centre = activity[:, :, HEIGHT // 4 - 1 : HEIGHT // 4 + 1, WIDTH // 4 - 1 : WIDTH // 4 + 1]
        out.append(centre.mean((2, 3)))
    return torch.cat(out).cpu()


def _offsets(angle_deg: float):
    yy, xx = torch.meshgrid(
        torch.arange(HEIGHT) - HEIGHT / 2 + 0.5,
        torch.arange(WIDTH) - WIDTH / 2 + 0.5,
        indexing="ij",
    )
    angle = math.radians(angle_deg)
    across = xx * math.cos(angle) + yy * math.sin(angle)
    along = -xx * math.sin(angle) + yy * math.cos(angle)
    return across, along, torch.sqrt(xx**2 + yy**2)


def _tuning(network: Hierarchy, make, sizes, feedback: bool) -> dict:
    preferred = _preferred_orientations(network)
    angles = sorted(set(preferred))
    frames = torch.cat([make(angle, size) for angle in angles for size in sizes])
    responses = _centre_responses(network, frames, feedback)
    per_stimulus = responses.shape[0] // (len(angles) * len(sizes))
    responses = responses.view(len(angles), len(sizes), per_stimulus, -1).mean(2)
    indices, curves = [], []
    for cell, angle in enumerate(preferred):
        curve = responses[angles.index(angle), :, cell].tolist()
        curves.append(curve)
        if max(curve) > 1e-4:
            indices.append(suppression_index(curve))
    return {
        "index": indices,
        "median": float(torch.tensor(indices).median()) if indices else 0.0,
        "fraction_above_0_3": sum(i > 0.3 for i in indices) / max(len(indices), 1),
        "responsive_cells": len(indices),
        "curves": curves,
    }


def size_tuning(
    network: Hierarchy, diameters=(4, 6, 8, 12, 16, 24, 32, 48), phases: int = 8
) -> dict:
    """Protocol 4. Suppression index of each cell, with and without top-down feedback."""

    def patch(angle, diameter):
        _, _, radius = _offsets(angle)
        mask = (radius <= diameter / 2).float()[None, None]
        gratings = torch.cat([grating(angle, 2 * math.pi * p / phases) for p in range(phases)])
        return 0.5 + mask * (gratings - 0.5)

    return {
        "with_feedback": _tuning(network, patch, diameters, feedback=True),
        "without_feedback": _tuning(network, patch, diameters, feedback=False),
    }


def end_stopping(network: Hierarchy, lengths=(2, 4, 6, 8, 12, 16, 24)) -> dict:
    """Protocol 5. End-stopping index of each cell, with and without top-down feedback."""

    def bar(angle, length):
        across, along, _ = _offsets(angle)
        mask = ((across.abs() <= 1.0) & (along.abs() <= length / 2)).float()
        return (0.5 + 0.5 * mask)[None, None]

    return {
        "with_feedback": _tuning(network, bar, lengths, feedback=True),
        "without_feedback": _tuning(network, bar, lengths, feedback=False),
    }


def _texture(height: int, width: int, seed: int) -> torch.Tensor:
    """Natural-statistics (1/f) texture, luminance ``[1, 1, H, W]`` in [0, 1]."""

    generator = torch.Generator().manual_seed(seed)
    image = torch.zeros(1, 1, height, width)
    for scale in (1, 2, 4, 8, 16):
        size = (max(2, height // scale), max(2, width // scale))
        noise = torch.rand(1, 1, *size, generator=generator)
        image += scale * F.interpolate(
            noise, size=(height, width), mode="bilinear", align_corners=False
        )
    image -= image.min()
    return image / image.max()


@torch.no_grad()
def predictive_remapping(engine, worlds: int = 8) -> dict:
    """Protocol 6. V1 activity before a saccade, carried through the movement's learned
    kernel (told) or the no-movement kernel (not told), against V1's actual activity
    after the saccade. Every activity is settled from rest the same way."""

    from .eye import COMMANDS, Eye

    network = engine.hierarchy
    level = network.levels[0]
    with_copy = [0.0] * len(COMMANDS)
    without = [0.0] * len(COMMANDS)
    opposite = [0.0] * len(COMMANDS)
    for seed in range(worlds):
        world = _texture(192, 256, seed).to(engine.device)
        for command in range(1, len(COMMANDS)):
            retina = Retina(WIDTH, HEIGHT, lagged=engine.retina.lagged, device=engine.device)
            eye = Eye(256, 192, fovea=(WIDTH, HEIGHT))
            view = eye.crop(world)
            retina.encode(view)  # a steady view: the lag holds the same image
            before = network.settle(retina.encode(view))[0]
            dx, dy = COMMANDS[command]
            eye.gaze = (eye.gaze[0] + dx * eye.saccade_px, eye.gaze[1] + dy * eye.saccade_px)
            after = network.settle(retina.encode(eye.crop(world)))[0]
            told = level.forecast(before, torch.tensor([command], device=engine.device))
            untold = level.forecast(before, torch.tensor([0], device=engine.device))
            # Direction control: the kernel of the opposite movement on the same
            # transition. A kernel that merely damps activity after any saccade
            # does as well with the wrong direction as with the right one.
            reverse = COMMANDS.index((-dx, -dy))
            wrong = level.forecast(before, torch.tensor([reverse], device=engine.device))
            with_copy[command] += float((told - after).square().mean())
            without[command] += float((untold - after).square().mean())
            opposite[command] += float((wrong - after).square().mean())
    per_direction = {
        str(COMMANDS[c]): with_copy[c] / without[c] if without[c] else float("nan")
        for c in range(1, len(COMMANDS))
    }
    total_without, total_opposite = sum(without), sum(opposite)
    return {
        "remapping_ratio": sum(with_copy) / total_without if total_without else float("nan"),
        "direction_specificity": sum(with_copy) / total_opposite
        if total_opposite
        else float("nan"),
        "per_direction": per_direction,
        "worlds": worlds,
    }


def _orienting_world(frames: int, seed: int = 0):
    """Left: texture panning predictably. Right: static texture with random flashes."""

    import numpy as np

    rng = np.random.default_rng(seed)
    pan = (_texture(192, 128 + 2 * frames, seed + 1)[0, 0].numpy() * 200 + 25).astype(np.uint8)
    still = (_texture(192, 128, seed + 2)[0, 0].numpy() * 200 + 25).astype(np.uint8)
    flash_until, flash = -1, (0, 0)
    for t in range(frames):
        frame = np.empty((192, 256), np.uint8)
        frame[:, :128] = pan[:, 2 * t : 2 * t + 128]
        frame[:, 128:] = still
        if t > flash_until and rng.random() < 1 / 15:
            flash_until, flash = t + 2, (int(rng.integers(0, 180)), int(rng.integers(128, 244)))
        if t <= flash_until:
            y, x = flash
            frame[y : y + 12, x : x + 12] = 255
        yield np.repeat(frame[..., None], 3, axis=2)


def _gaze_statistics(decisions: list[dict]) -> dict:
    """The plan's statistic and inhibition of return, from an eye's decision log.

    chosen_surprise_percentile: where the chosen target's surprise ranks among the
    possible targets (0 lowest, 1 highest; a random choice averages 0.5). The eye is
    built to seek surprise, so this is a sanity check, not a learned phenomenon.
    revisit_rate vs random_revisit_rate: how often a saccade returns to a recent
    fixation, against the rate a random choice among the same targets would give.
    """

    percentiles, revisits, chance = [], [], []
    for decision in decisions:
        raw, revisit = decision["raw"], decision["revisit"]
        if not raw:
            continue
        chance.append(sum(revisit.values()) / len(revisit))
        chosen = decision["chosen"]
        if not chosen:
            continue
        revisits.append(float(revisit[chosen]))
        if len(raw) > 1:
            below = sum(value < raw[chosen] for value in raw.values())
            percentiles.append(below / (len(raw) - 1))

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    return {
        "chosen_surprise_percentile": mean(percentiles),
        "revisit_rate": mean(revisits),
        "random_revisit_rate": mean(chance),
        "saccades": len(revisits),
    }


def habituation_of_orienting(
    engine, frames: int = 900, first: int = 200, last: int = 300, seeds: int = 3
) -> dict:
    """Protocol 7. Dwell of the gaze on the unpredictable-onset half, with the
    periphery learning (habituation possible) and frozen (no habituation), averaged
    over seeds. Dwell is measured instead of saccade direction counts: those are
    bounded by where the eye stands, so an eye that reached the onsets could only
    move back."""

    report = {}
    for condition, learn in (("learning", True), ("frozen", False)):
        first_dwell, last_dwell, mean_x, decisions = [], [], [], []
        for seed in range(seeds):
            subject = copy.deepcopy(engine)
            subject.learn = learn
            subject.eyes = []
            subject.reset()
            positions = []
            for frame in _orienting_world(frames, seed):
                subject.predict(frame)
                positions.append(subject.eyes[0].gaze[0])
            centre = 128
            first_dwell.append(sum(x > centre for x in positions[:first]) / first)
            last_dwell.append(sum(x > centre for x in positions[-last:]) / last)
            mean_x.append(sum(positions[-last:]) / last)
            decisions.extend(subject.eyes[0].decisions)
        report[condition] = {
            "dwell_on_onsets_first": sum(first_dwell) / seeds,
            "dwell_on_onsets_last": sum(last_dwell) / seeds,
            "mean_gaze_x_last": sum(mean_x) / seeds,
            "per_seed_dwell_last": last_dwell,
            "gaze": _gaze_statistics(decisions),
        }
    return report


@torch.no_grad()
def motion_cells(cortex, texture_seeds=range(10), frames_grating: int = 16) -> dict:
    """Protocol 16. Direction selectivity of MT's types on drifting gratings, the
    compass directions their learned displacements cover, and the direction error
    of the decoded velocity on translating textures."""

    level, directions, period = cortex.fine, 12, 8.0
    rows, cols = slice(HEIGHT // 4 - 6, HEIGHT // 4 + 6), slice(WIDTH // 4 - 8, WIDTH // 4 + 8)
    curves = torch.zeros(2, directions, level.types)
    for s, speed in enumerate((1.0, 2.0)):
        for d in range(directions):
            angle = 360.0 * d / directions
            cortex.reset()
            seen = []
            for t in range(frames_grating):
                frame = grating(angle, -2 * math.pi * speed * t / period, period)
                cortex.step(frame, learn=False)
                if t >= 3:
                    before = cortex._last_pair
                    r = level.responsibilities(*before)
                    seen.append(r[0, :, rows, cols].mean((1, 2)).cpu())
            curves[s, d] = torch.stack(seen).mean(0)
    dsi = []
    for m in range(level.types):
        speed = int(curves[:, :, m].max(1).values.argmax())
        curve = curves[speed, :, m]
        best = int(curve.argmax())
        if float(curve[best]) < 0.05:
            continue
        dsi.append(direction_index(float(curve[best]), float(curve[(best + 6) % 12])))
    displacements = level.displacements().cpu()
    long = displacements[displacements.norm(dim=1) >= 0.5]
    sectors = {
        int(((math.degrees(math.atan2(float(dy), float(dx))) + 22.5) % 360) // 45)
        for dx, dy in long
    }
    errors = []
    for seed in texture_seeds:
        image = _texture(96 + 100, 128 + 100, seed)
        for k in range(8):
            angle = 2 * math.pi * k / 8
            for speed in (1.0, 2.0, 4.0):
                cortex.reset()
                decoded = []
                for t in range(20):
                    dx, dy = speed * t * math.cos(angle), speed * t * math.sin(angle)
                    x0, y0 = int(round(50 - dx)), int(round(50 - dy))
                    x0, y0 = min(max(x0, 0), 100), min(max(y0, 0), 100)
                    v = cortex.step(image[..., y0 : y0 + 96, x0 : x0 + 128], learn=False)
                    if v is not None and t >= 3:
                        decoded.append(v[0, :, 24:72, 32:96].mean((1, 2)).cpu())
                vx, vy = torch.stack(decoded).mean(0).tolist()
                error = math.degrees(math.atan2(vy, vx) - angle)
                errors.append(abs((error + 180) % 360 - 180))
    # Amended criterion: does a type's preferred grating direction agree with the
    # direction of its own learned displacement?
    agreeing, judged = 0, 0
    for m in range(level.types):
        dx, dy = displacements[m].tolist()
        if math.hypot(dx, dy) < 0.5:
            continue
        speed = int(curves[:, :, m].max(1).values.argmax())
        if float(curves[speed, :, m].max()) < 0.05:
            continue
        judged += 1
        preferred = 360.0 * int(curves[speed, :, m].argmax()) / directions
        learned = math.degrees(math.atan2(dy, dx))
        agreeing += abs((preferred - learned + 180) % 360 - 180) <= 45
    return {
        "dsi": dsi,
        "fraction_dsi_over_half": sum(v > 0.5 for v in dsi) / level.types,
        "types_judged": judged,
        "tuning_agreement": agreeing / judged if judged else 0.0,
        "responsive_types": len(dsi),
        "sectors_covered": len(sectors),
        "displacements": displacements.tolist(),
        "median_direction_error": float(torch.tensor(errors).median()),
        "curves": curves.tolist(),
    }
