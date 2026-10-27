import json

import numpy as np

from cortical_mirror.scene.graph import SceneGraph
from cortical_mirror.scene.world import SpriteWorld


def watch(world: SpriteWorld, frames: int, graph: SceneGraph | None = None) -> SceneGraph:
    graph = graph or SceneGraph(world.width, world.height)
    for _ in range(frames):
        world.step()
        graph.observe(world.render())
    return graph


def matched(graph: SceneGraph, thing) -> object:
    return min(graph.nodes, key=lambda n: (n.x - thing.x) ** 2 + (n.y - thing.y) ** 2)


def test_a_static_world_has_no_objects():
    world = SpriteWorld(seed=0, objects=2, speed=(0.0, 0.0))
    graph = watch(world, 40)
    assert graph.nodes == []


def test_one_moving_object_becomes_one_node_that_follows_it():
    world = SpriteWorld(seed=1, objects=1)
    graph = watch(world, 80)
    assert len(graph.nodes) == 1
    node, thing = graph.nodes[0], world.objects[0]
    assert abs(node.x - thing.x) <= 3 and abs(node.y - thing.y) <= 3


def test_each_of_several_objects_gets_its_own_node():
    world = SpriteWorld(seed=5, objects=3)
    graph = watch(world, 150)
    assert len(graph.nodes) == 3
    assert len({matched(graph, thing).id for thing in world.objects}) == 3


def test_rendering_the_graph_reproduces_what_is_seen():
    world = SpriteWorld(seed=2, objects=2)
    graph = watch(world, 150)
    error = float(np.mean((graph.render() - world.render()) ** 2))
    assert error < 0.002


def test_deleting_a_node_shows_the_remembered_background_behind_it():
    world = SpriteWorld(seed=1, objects=1)
    graph = watch(world, 120)
    region = world.truth()[0]["full"]
    gone = graph.edit(delete={graph.nodes[0].id})
    error = float(np.mean((gone[region] - world.background_view()[region]) ** 2))
    assert error < 0.003


def test_editing_changes_a_copy_and_leaves_the_graph_as_it_was():
    world = SpriteWorld(seed=1, objects=1)
    graph = watch(world, 80)
    node = graph.nodes[0]
    before = (node.x, node.y, graph.render().copy())
    moved = graph.edit(move={node.id: (30, 0)})
    assert not np.allclose(moved, before[2])
    assert (graph.nodes[0].x, graph.nodes[0].y) == before[:2]
    assert np.array_equal(graph.render(), before[2])


def test_the_nearer_of_two_crossing_objects_ends_up_in_front():
    world = SpriteWorld(seed=7, objects=2, speed=(0.0, 0.0))
    back, front = world.objects
    back.x, back.y, back.vx = 60.0, 96.0, 1.5
    front.x, front.y, front.vx = 196.0, 100.0, -1.5
    graph = watch(world, 110)
    assert matched(graph, front).depth > matched(graph, back).depth


def test_the_graph_is_served_as_json_with_nodes_and_edges():
    world = SpriteWorld(seed=5, objects=3)
    graph = watch(world, 150)
    data = json.loads(json.dumps(graph.to_json()))
    assert {"nodes", "edges", "self"} <= data.keys()
    node = data["nodes"][0]
    assert {"id", "kind", "x", "y", "vx", "vy", "depth", "state", "belief"} <= node.keys()
    assert {edge["relation"] for edge in data["edges"]} <= {"occludes", "moves_with", "near"}


def occlusion_world(seed: int = 0) -> SpriteWorld:
    """One small object passing behind a wide screen in the middle."""

    world = SpriteWorld(seed=seed, objects=1, occluders=1, occluder_width=44, radius=(7, 9))
    thing = world.objects[0]
    thing.x, thing.y, thing.vx, thing.vy = 40.0, 96.0, 1.5, 0.0
    return world


def test_a_graph_that_forgets_at_once_loses_a_hidden_object_and_a_patient_one_keeps_it():
    forgetful = SceneGraph(persistence=1.0)
    patient = SceneGraph(persistence=60.0)
    for graph in (forgetful, patient):
        world = occlusion_world()
        for _ in range(58):
            world.step()
            graph.observe(world.render())
        hidden = world.truth()[0]["visible"].sum() == 0
        assert hidden
        thing = world.objects[0]
        believed = [n for n in graph.nodes if abs(n.x - thing.x) < 6 and abs(n.y - thing.y) < 6]
        if graph is forgetful:
            assert believed == []
        else:
            assert len(believed) == 1 and believed[0].state == "occluded"


def test_expect_moves_every_node_on_by_its_velocity_without_changing_the_graph():
    world = SpriteWorld(seed=1, objects=1)
    graph = watch(world, 80)
    node = graph.nodes[0]
    before = (node.x, node.y)
    expected = graph.expect()
    assert (node.x, node.y) == before
    nudged = graph.edit(move={node.id: (node.vx, node.vy)})
    assert np.allclose(expected, nudged)


def test_recognising_a_forgotten_object_teaches_the_graph_to_keep_things_longer():
    graph = SceneGraph(persistence=1.0, learn_persistence=True)
    for seed in range(3):
        graph.reset()
        world = occlusion_world(seed)
        for _ in range(150):
            world.step()
            graph.observe(world.render())
    assert graph.persistence > 5.0


def test_objects_that_vanish_for_good_teach_it_to_forget_sooner():
    graph = SceneGraph(persistence=40.0, learn_persistence=True)
    for seed in range(4):
        graph.reset()
        world = occlusion_world(seed)
        for t in range(400):
            world.step()
            if world.objects and world.truth()[0]["visible"].sum() == 0 and t > 30:
                world.objects.clear()
            graph.observe(world.render())
    assert graph.persistence < 40.0


def test_learned_persistence_survives_saving_and_a_new_scene(tmp_path):
    graph = SceneGraph(persistence=7.5, learn_persistence=True)
    graph.save(tmp_path / "scene.pt")
    loaded = SceneGraph.load(tmp_path / "scene.pt")
    assert loaded.persistence == 7.5 and loaded.learn_persistence
    loaded.reset()
    assert loaded.persistence == 7.5
