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


def test_stillness_baseline_isolates_what_the_temporal_predictor_learned():
    newborn = Hierarchy(2, [LevelConfig(temporal_rate=0.0)])
    still = run(newborn, pan(30, 2))
    # A predictor that never learned expects nothing to change: identical errors.
    assert all(abs(r.code_error - r.stillness_error) < 1e-9 for r in still[2:])
    learner = Hierarchy(2, [LevelConfig()])
    results = run(learner, pan(3000, 2))
    model = sum(r.code_error for r in results[-200:])
    stillness = sum(r.stillness_error for r in results[-200:])
    assert model < 0.8 * stillness


def test_newborn_on_a_still_scene_keeps_diverse_living_cells():
    # A single stream of one unchanging image is the worst case: every frame
    # pushes the same update. Cells must stay different and keep firing.
    from tests.cortex_fixtures import texture

    network = Hierarchy(2, [LevelConfig()])
    still = texture(48, 64, seed=5)
    results = run(network, [still] * 150)
    weights = network.levels[0].W.flatten(1)
    weights = weights / weights.norm(dim=1, keepdim=True)
    similarity = weights @ weights.T
    off_diagonal = similarity[~torch.eye(len(similarity), dtype=bool)].abs().mean()
    assert float(off_diagonal) < 0.5
    assert float((results[-1].activities[0] > 0).float().mean()) > 0.005
