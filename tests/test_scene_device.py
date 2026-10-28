import numpy as np
import pytest
import torch

from cortical_mirror.scene.graph import SceneGraph
from cortical_mirror.scene.world import CreatureWorld

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def play(graph: SceneGraph, frames: int = 120):
    world = CreatureWorld(seed=2, creatures=2, occluders=1, occluder_width=40)
    for _ in range(frames):
        world.step()
        graph.observe(world.render())
    return graph


@pytest.mark.parametrize("device", DEVICES)
def test_the_search_on_a_torch_device_finds_what_the_arrays_find(device):
    arrays, tensors = play(SceneGraph()), play(SceneGraph(device=device))
    assert [n.id for n in arrays.nodes] == [n.id for n in tensors.nodes]
    for a, b in zip(arrays.nodes, tensors.nodes, strict=True):
        assert abs(a.x - b.x) < 1e-3 and abs(a.y - b.y) < 1e-3
    assert np.allclose(arrays.render(), tensors.render(), atol=1e-4)
