import json
import random

import numpy as np

from cortical_mirror.scene.eye import SACCADE, EyeGraph, look
from cortical_mirror.scene.world import SpriteWorld
from cortical_mirror.cortex.eye import COMMANDS

WORLD = (384, 288)
VIEW = (256, 192)


def still_world(seed: int = 0, objects: int = 0) -> SpriteWorld:
    return SpriteWorld(seed=seed, width=WORLD[0], height=WORLD[1], objects=objects)


def settle(graph: EyeGraph, world: SpriteWorld, gaze, frames: int = 20) -> None:
    for _ in range(frames):
        world.step()
        graph.observe(look(world.render(), gaze, VIEW), 0)


def saccade(gaze, command):
    dx, dy = COMMANDS[command]
    return gaze[0] + dx * SACCADE, gaze[1] + dy * SACCADE


def test_looking_crops_the_view_around_the_gaze():
    world = still_world()
    image = world.render()
    view = look(image, (192, 144), VIEW)
    assert view.shape == (192, 256)
    assert np.array_equal(view, image[48:240, 64:320])


def test_a_newborn_sees_the_world_jump_when_its_own_eye_moves():
    world, graph, gaze = still_world(), EyeGraph(VIEW), (192, 144)
    settle(graph, world, gaze)
    gaze = saccade(gaze, 1)
    graph.observe(look(world.render(), gaze, VIEW), 1)
    assert abs(graph.world_shift[0] + SACCADE) <= 2 and abs(graph.world_shift[1]) <= 2
    assert graph.gaze == graph.start


def eye_trained(frames: int = 600, seed: int = 0):
    rng = random.Random(seed)
    world, graph, gaze = still_world(seed), EyeGraph(VIEW), (192, 144)
    settle(graph, world, gaze)
    for t in range(frames):
        command = 0
        if t % 5 == 0:
            valid = [
                c
                for c in range(1, len(COMMANDS))
                if 128 <= saccade(gaze, c)[0] <= 256 and 96 <= saccade(gaze, c)[1] <= 192
            ]
            command = rng.choice(valid)
            gaze = saccade(gaze, command)
        graph.observe(look(world.render(), gaze, VIEW), command)
    return graph, world, gaze


def test_the_motor_map_learns_how_far_each_command_moves_the_view():
    graph, _, _ = eye_trained()
    for command, (dx, dy) in enumerate(COMMANDS):
        assert np.allclose(graph.motor[command], (dx * SACCADE, dy * SACCADE), atol=2.0)


def test_with_its_motor_map_learned_its_own_saccades_leave_the_world_still():
    graph, world, gaze = eye_trained()
    shift = graph.world_shift.copy()
    command = 5 if gaze[0] > 192 else 1
    graph.observe(look(world.render(), saccade(gaze, command), VIEW), command)
    assert np.abs(graph.world_shift - shift).max() <= 2


def test_the_same_shift_without_a_command_is_the_world_moving():
    graph, world, gaze = eye_trained()
    shift = graph.world_shift.copy()
    command = 5 if gaze[0] > 192 else 1
    graph.observe(look(world.render(), saccade(gaze, command), VIEW), 0)
    moved = graph.world_shift - shift
    assert abs(abs(moved[0]) - SACCADE) <= 2 and abs(moved[1]) <= 2


def test_an_object_keeps_its_place_in_the_world_across_a_learned_saccade():
    graph, _, gaze = eye_trained()
    world = SpriteWorld(seed=9, width=WORLD[0], height=WORLD[1], objects=1, speed=(0.0, 0.0))
    thing = world.objects[0]
    thing.x, thing.y, thing.vx = gaze[0] - 60.0, gaze[1], 1.0
    graph.reset_scene()
    settle(graph, world, gaze, 60)
    before = graph.world_positions()
    command = 1 if gaze[0] < 256 else 5
    world.step()
    graph.observe(look(world.render(), saccade(gaze, command), VIEW), command)
    after = graph.world_positions()
    assert len(before) == 1 and len(after) == 1
    (_, bx, by), (_, ax, ay) = before[0], after[0]
    assert abs(ax - bx - 1.0) <= 3 and abs(ay - by) <= 3


def test_the_self_node_is_part_of_the_graph():
    graph, _, _ = eye_trained(frames=30)
    data = json.loads(json.dumps(graph.to_json()))
    eye = data["self"]
    assert eye["kind"] == "eye" and len(eye["gaze"]) == 2 and len(eye["motor"]) == len(COMMANDS)
