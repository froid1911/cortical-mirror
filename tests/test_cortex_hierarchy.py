import pytest
import torch

from cortical_mirror.cortex.hierarchy import Hierarchy, LevelConfig
from cortical_mirror.cortex.retina import Retina
from tests.cortex_fixtures import pan, scenes


def run(network: Hierarchy, frames, learn=True):
    retina = Retina()
    return [network.step(retina.encode(frame), learn=learn) for frame in frames]


def test_cortex_never_uses_autograd(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the cortex must not use backpropagation")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    network = Hierarchy(2, [LevelConfig()])
    results = run(network, scenes(50))
    tensors = [t for t in network.state_dict()["weights"].values()] + results[-1].activities
    assert all(not t.requires_grad and t.grad_fn is None for t in tensors)


def test_feedback_weights_converge_without_copying():
    network = Hierarchy(2, [LevelConfig()])
    level = network.levels[0]
    start = float((level.B - level.W).norm() / level.W.norm())
    run(network, scenes(2000))
    end = float((level.B - level.W).norm() / level.W.norm())
    assert start > 0.5
    assert end < 0.1


def test_hierarchy_long_run_stays_bounded():
    network = Hierarchy(2, [LevelConfig()])
    results = run(network, scenes(6000))
    recent = torch.stack(
        [(r.activities[0] > 0).float().mean((0, 2, 3)) for r in results[-500:]]
    ).mean(0)
    assert torch.isfinite(network.levels[0].W).all() and torch.isfinite(network.levels[0].V).all()
    assert float(recent.min()) >= 0.005
    assert float(recent.max()) <= 0.5


def test_temporal_predictor_beats_persistence_on_a_pan():
    network = Hierarchy(2, [LevelConfig()])
    results = run(network, pan(3000, 2))
    model = sum(r.code_error for r in results[-200:])
    persistence = sum(r.persistence_error for r in results[-200:])
    assert model < 0.8 * persistence


def test_checkpoint_architecture_mismatch_is_reported():
    phase1 = Hierarchy(2, [LevelConfig()])
    phase2 = Hierarchy(4, [LevelConfig()])
    with pytest.raises(ValueError, match="architecture"):
        phase2.load_state_dict(phase1.state_dict())
