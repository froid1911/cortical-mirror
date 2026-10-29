import numpy as np

from cortical_mirror.cortex.learned_graph import LearnedGraph
from cortical_mirror.cortex.mt import MotionCortex
from cortical_mirror.scene.world import SpriteWorld


def watched(frames: int = 6):
    world = SpriteWorld(seed=3)
    graph = LearnedGraph(cortex=MotionCortex(types=4, seed=0, fast=True))
    for _ in range(frames):
        world.step()
        graph.observe(world.render())
    return world, graph


def test_a_learned_graph_speaks_the_scene_graph_s_language():
    world, graph = watched()
    scene = graph.to_json()
    assert {"frame", "nodes", "edges", "groups", "self"} <= set(scene)
    assert scene["frame"] == 6
    for node in scene["nodes"]:
        assert {"id", "kind", "x", "y", "vx", "vy", "depth", "state", "belief", "size"} <= set(node)


def test_its_pictures_have_the_world_s_size():
    world, graph = watched()
    assert graph.render().shape == (world.height, world.width)
    assert graph.edit(delete=set(), move={}).shape == (world.height, world.width)


def test_its_nodes_have_masks_and_can_be_edited():
    _, graph = watched()
    graph.scene.labels = np.zeros_like(graph.scene.labels)
    graph.scene.labels[10:14, 20:24] = 1
    graph._refresh()
    assert [n.id for n in graph.nodes] == [1]
    node = graph.nodes[0]
    assert graph._alpha(node).sum() == 16 * 16
    frame = graph.scene.frame
    moved = graph.edit(move={1: (30.0, 0.0)})
    assert not np.allclose(moved, frame)
    copied = graph.edit(duplicate={1: (30.0, 0.0)})
    # A copy leaves the original where it is.
    assert np.allclose(copied[40:56, 80:96], frame[40:56, 80:96])


def test_a_reset_forgets_the_scene():
    _, graph = watched()
    graph.reset()
    assert graph.nodes == [] and graph.frame_count == 0


def test_the_expectation_moves_each_object_by_its_velocity():
    _, graph = watched()
    frame = graph.scene.frame.copy()
    frame[40:56, 80:96] = 0.95
    graph.scene.frame = frame
    graph.scene.memory[:] = 0.2
    graph.scene.seen[:] = 1
    graph.scene.labels = np.zeros_like(graph.scene.labels)
    graph.scene.labels[10:14, 20:24] = 1
    graph.slots.u[1] = 0.0
    graph.slots.u[1, 0] = 2.0  # 2 px of the 128-wide view = 4 image px per frame
    graph._refresh()
    expected = graph.expect()
    assert np.allclose(expected[40:56, 84:100], 0.95, atol=1e-4)
    assert np.allclose(expected[40:56, 80:84], 0.2, atol=1e-4)
    # Elsewhere the last frame stands.
    assert np.allclose(expected[100:, :], frame[100:, :])


def test_without_motion_the_expectation_is_the_last_frame():
    _, graph = watched()
    for k in range(len(graph.slots.u)):
        graph.slots.u[k] = 0.0
    graph._refresh()
    assert np.allclose(graph.expect(), graph.scene.frame, atol=1e-4)
