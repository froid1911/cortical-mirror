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
    level = network.levels[0]
    saved_step = level.step_size
    level.step_size = 0.9 / max(level.stability_limit(code.shape), 1e-6)
    start = torch.zeros(
        len(stimuli), level.config.cells, HEIGHT // 2, WIDTH // 2, device=network.device
    )
    activity = level.settle(code, start, iterations=30)
    level.step_size = saved_step
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
    level = network.levels[0]
    saved_step = level.step_size
    level.step_size = min(0.9 / max(level.stability_limit(code.shape), 1e-6), 0.1)
    start = torch.zeros(
        len(code), level.config.cells, HEIGHT // 2, WIDTH // 2, device=network.device
    )
    activity = level.settle(code, start, iterations=30)
    level.step_size = saved_step
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
