from collections import Counter

import numpy as np

from cortical_mirror.cortex.eye import COMMANDS
from cortical_mirror.scene.attention import FEATURES, Attention
from cortical_mirror.scene.eye import SACCADE, EyeGraph, look
from cortical_mirror.scene.world import SpriteWorld

VIEW = (256, 192)
TRUE_MOTOR = np.array(COMMANDS, float) * SACCADE


def graph_after(frames: int = 20, fovea: int | None = 96) -> tuple[EyeGraph, SpriteWorld]:
    world = SpriteWorld(seed=0, width=384, height=288, objects=2)
    graph = EyeGraph(VIEW, motor=TRUE_MOTOR, learn_motor=False, fovea=fovea)
    for _ in range(frames):
        world.step()
        graph.observe(look(world.render(), (192, 144), VIEW), 0)
    return graph, world


def test_only_the_fovea_is_seen_sharply():
    graph, world = graph_after(1)
    view = look(world.render(), (192, 144), VIEW)
    seen = graph.foveate(view)
    centre = (slice(96 - 30, 96 + 30), slice(128 - 30, 128 + 30))
    assert np.allclose(seen[centre], view[centre])
    edge = (slice(0, 30), slice(0, 30))
    assert np.abs(np.diff(seen[edge], axis=1)).mean() < np.abs(np.diff(view[edge], axis=1)).mean()


def test_features_describe_every_command_between_zero_and_one():
    graph, _ = graph_after()
    features = Attention().features(graph)
    assert features.shape == (len(COMMANDS), len(FEATURES))
    assert features.min() >= 0.0 and features.max() <= 1.0


def test_a_newborn_looks_at_random():
    graph, _ = graph_after()
    attention = Attention(seed=1)
    choices = Counter(attention.choose(graph, allowed=range(1, 9)) for _ in range(800))
    assert set(choices) == set(range(1, 9))
    assert max(choices.values()) < 2 * min(choices.values())


def test_a_reflex_for_surprise_looks_where_the_error_is():
    graph, _ = graph_after()
    attention = Attention(policy="surprise")
    features = attention.features(graph)
    error = FEATURES.index("error")
    chosen = attention.choose(graph, allowed=range(1, 9))
    assert features[chosen, error] == features[1:, error].max()


def test_progress_teaches_the_weight_of_the_feature_that_predicted_it():
    attention = Attention(learn=True)
    rng = np.random.default_rng(0)
    unknown = FEATURES.index("unknown")
    for _ in range(300):
        features = rng.random(len(FEATURES))
        attention.learn(features, progress=0.5 * features[unknown])
    assert attention.weights[unknown] > 0.3
    others = [i for i, name in enumerate(FEATURES) if name not in ("unknown", "bias")]
    assert np.abs(attention.weights[others]).max() < 0.15


def test_a_fixation_records_how_much_the_graph_learned_there():
    graph, world = graph_after(5)
    attention = Attention()
    for t in range(40):
        world.step()
        command = attention.step(graph, t, allowed=lambda c: True)
        graph.observe(look(world.render(), (192, 144), VIEW), 0 * command)
    assert len(attention.history) >= 3
    assert all(np.isfinite(h["progress"]) for h in attention.history)
