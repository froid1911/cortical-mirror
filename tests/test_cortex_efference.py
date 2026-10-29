import torch

from cortical_mirror.cortex.efference import Efference


def test_a_newborn_map_expects_the_same_from_every_command():
    efference = Efference(commands=9, types=4)
    displacements = torch.tensor([[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -2.0]])
    for command in range(9):
        assert efference.self_velocity(command, displacements).abs().max() < 1e-6


def test_a_command_always_followed_by_one_motion_comes_to_expect_it():
    efference = Efference(commands=9, types=4, rate=0.2)
    displacements = torch.tensor([[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -2.0]])
    r = torch.zeros(1, 4, 6, 8)
    r[:, 0] = 1.0  # after command 1 the "right" type explains everything
    for _ in range(50):
        efference.learn(1, r)
    assert torch.allclose(
        efference.self_velocity(1, displacements), torch.tensor([2.0, 0.0]), atol=0.05
    )
    world = efference.world_motion(
        torch.full((1, 2, 6, 8), 0.0) + torch.tensor([2.0, 0.0])[None, :, None, None],
        1,
        displacements,
    )
    assert world.abs().max() < 0.05
    assert efference.self_velocity(5, displacements).abs().max() < 1e-6


def test_protocol_18_reports_commanded_and_uncommanded_world_motion():
    from cortical_mirror.cortex.mt import MotionCortex
    from cortical_mirror.scene import phenomena

    mt = MotionCortex(types=4, seed=0, settle=5)
    efference = phenomena.train_efference(mt, seeds=[800], frames=30)
    report = phenomena.self_world_learned(mt, efference, seeds=[8000], settle=5)
    assert report["commanded"] >= 0 and report["uncommanded"] >= 0
    assert 0 <= report["sensed_direction_error"] <= 180
