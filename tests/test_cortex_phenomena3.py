import pytest
import torch

from cortical_mirror.cortex.engine import CortexEngine
from cortical_mirror.cortex.eye import COMMANDS
from cortical_mirror.cortex.phenomena import habituation_of_orienting, predictive_remapping


def test_untrained_remapping_ratio_is_one():
    result = predictive_remapping(CortexEngine(phase=3), worlds=2)
    assert result["remapping_ratio"] == pytest.approx(1.0)


def shift_kernels(engine: CortexEngine, sign: int) -> CortexEngine:
    """Fixture: kernels that move V1 activity as a saccade moves the image (8 retinal
    pixels = 4 V1 positions, opposite to the eye); sign -1 moves it the wrong way."""

    level = engine.hierarchy.levels[0]
    cells = level.config.cells
    for command, (dx, dy) in enumerate(COMMANDS):
        kernel = torch.zeros(cells, cells, 9, 9)
        kernel[torch.arange(cells), torch.arange(cells), 4 + sign * 4 * dy, 4 + sign * 4 * dx] = 1.0
        level.V[command] = kernel
    return engine


def test_remapping_rewards_kernels_that_shift_the_right_way():
    right = predictive_remapping(shift_kernels(CortexEngine(phase=3), 1), worlds=2)
    wrong = predictive_remapping(shift_kernels(CortexEngine(phase=3), -1), worlds=2)
    assert right["remapping_ratio"] < 0.95
    assert wrong["remapping_ratio"] > 1.2


def test_habituation_measure_reports_saccade_fractions():
    result = habituation_of_orienting(CortexEngine(phase=3), frames=120, first=60, last=60)
    for condition in ("learning", "frozen"):
        for window in ("first", "last"):
            fraction = result[condition][window]["toward_onsets"]
            assert fraction is None or 0.0 <= fraction <= 1.0
    assert result["learning"]["saccades"] > 0
