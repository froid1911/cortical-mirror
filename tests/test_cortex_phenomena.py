import math

import pytest
import torch

from cortical_mirror.cortex.hierarchy import Hierarchy, LevelConfig
from cortical_mirror.cortex.phenomena import (
    mismatch_indices,
    oddball_adaptation,
    orientation_tuning,
)


def bar_cells() -> Hierarchy:
    """Test fixture: V1 cells hand-set to oriented bars (ON centre stripe, OFF flanks)."""

    network = Hierarchy(2, [LevelConfig()])
    level = network.levels[0]
    yy, xx = torch.meshgrid(torch.arange(8.0) - 3.5, torch.arange(8.0) - 3.5, indexing="ij")
    kernels = []
    for cell in range(32):
        angle = math.pi * (cell % 12) / 12
        distance = xx * math.cos(angle) + yy * math.sin(angle)
        profile = torch.cos(2 * math.pi * distance / 8) * torch.exp(-(xx**2 + yy**2) / 18)
        kernels.append(torch.stack((profile.clamp_min(0), (-profile).clamp_min(0))))
    weights = torch.stack(kernels)
    weights = weights / weights.flatten(1).norm(dim=1)[:, None, None, None]
    level.W, level.B = weights.clone(), weights.clone()
    return network


def test_oriented_bar_cells_have_high_osi():
    result = orientation_tuning(bar_cells())
    assert result["median_osi"] > 0.5
    assert result["responsive_cells"] == 32


def test_random_cells_have_low_osi():
    assert orientation_tuning(Hierarchy(2, [LevelConfig()]))["median_osi"] < 0.4


def test_mismatch_indices_math():
    result = mismatch_indices(deviant=3.0, standard=1.0, control=2.0)
    norm = math.sqrt(9 + 1 + 4)
    assert result["iRS"] == pytest.approx(1 / norm)
    assert result["iPE"] == pytest.approx(1 / norm)
    assert result["iMM"] == pytest.approx(2 / norm)


def test_oddball_adaptation_leaves_the_network_unchanged():
    network = Hierarchy(2, [LevelConfig()])
    before = network.levels[0].W.clone()
    result = oddball_adaptation(network, trials=100)
    assert torch.equal(network.levels[0].W, before) and network.age_frames == 0
    assert all(math.isfinite(result[key]) for key in ("DEV", "STD", "CTR", "iRS", "iPE", "iMM"))
