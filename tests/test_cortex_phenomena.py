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


def lagged_cells(shift: int) -> Hierarchy:
    """Fixture: each cell's lagged weights are its current weights moved `shift` pixels
    along the cell's own preferred axis (perpendicular to its stripes)."""

    network = Hierarchy(4, [LevelConfig()])
    current = bar_cells().levels[0].W
    lagged = torch.empty_like(current)
    for cell in range(len(current)):
        angle = math.pi * (cell % 12) / 12
        dx, dy = round(shift * math.cos(angle)), round(shift * math.sin(angle))
        lagged[cell] = torch.roll(current[cell], shifts=(dy, dx), dims=(1, 2))
    weights = torch.cat((current, lagged), dim=1)
    weights = weights / weights.flatten(1).norm(dim=1)[:, None, None, None]
    network.levels[0].W, network.levels[0].B = weights.clone(), weights.clone()
    return network


def test_dsi_math():
    from cortical_mirror.cortex.phenomena import direction_index

    assert direction_index(preferred=3.0, opposite=1.0) == pytest.approx(0.5)
    assert direction_index(preferred=1.0, opposite=1.0) == 0.0


def test_mirror_symmetric_cells_have_low_dsi():
    from cortical_mirror.cortex.phenomena import direction_selectivity

    assert direction_selectivity(lagged_cells(0))["median_dsi"] < 0.1


def test_shifted_lag_cells_are_direction_selective():
    from cortical_mirror.cortex.phenomena import direction_selectivity

    result = direction_selectivity(lagged_cells(2))
    assert result["fraction_selective"] > 0.3


def test_suppression_index_math():
    from cortical_mirror.cortex.phenomena import suppression_index

    assert suppression_index([1.0, 3.0, 2.0]) == pytest.approx(1 / 3)
    assert suppression_index([1.0, 2.0, 3.0]) == 0.0
    assert suppression_index([0.0, 0.0]) == 0.0
