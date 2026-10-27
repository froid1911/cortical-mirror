import torch

from cortical_mirror.cortex.eye import COMMANDS, Eye
from cortical_mirror.cortex.hierarchy import Hierarchy, LevelConfig
from tests.cortex_fixtures import texture


def world() -> torch.Tensor:
    return texture(192, 256, seed=4)


def test_eye_stays_inside_the_world():
    eye = Eye(period_frames=1)
    surprise = torch.zeros(1, 1, 48, 64)
    surprise[..., 0, 0] = 1.0  # pull toward the top-left corner
    for _ in range(100):
        fovea, _ = eye.look(world(), surprise)
        x, y = eye.gaze
        assert fovea.shape == (1, 1, 96, 128)
        assert 64 <= x <= 256 - 64 and 48 <= y <= 192 - 48


def test_saccade_goes_toward_surprise():
    eye = Eye(period_frames=1)
    surprise = torch.zeros(1, 1, 48, 64)
    surprise[..., 24, 63] = 1.0  # far to the right, at the same height
    _, command = eye.look(world(), surprise)
    assert COMMANDS[command] == (1, 0)
    assert eye.gaze[0] > 128


def test_inhibition_of_return_avoids_last_location():
    eye = Eye(period_frames=1, inhibition=1.0)
    flat = torch.ones(1, 1, 48, 64)  # nothing is more surprising than anything else
    visited = [eye.gaze]
    for _ in range(6):
        eye.look(world(), flat)
        visited.append(eye.gaze)
    assert len(set(visited)) == len(visited)


def test_command_selects_its_own_temporal_kernel():
    network = Hierarchy(2, [LevelConfig(motor_commands=len(COMMANDS))])
    level = network.levels[0]
    assert level.V.shape[0] == len(COMMANDS)
    level.V[3].zero_()
    activity = torch.rand(1, 32, 24, 32)
    assert torch.equal(level.forecast(activity, torch.tensor([3])), torch.zeros_like(activity))
    assert not torch.equal(level.forecast(activity, torch.tensor([0])), torch.zeros_like(activity))


def test_phase_one_checkpoints_still_load():
    old = Hierarchy(2, [LevelConfig()]).state_dict()
    for level in old["architecture"]["levels"]:
        level.pop("motor_commands", None)  # written before the field existed
    Hierarchy(2, [LevelConfig()]).load_state_dict(old)
