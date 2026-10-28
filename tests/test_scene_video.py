import numpy as np

from cortical_mirror.scene.eye import look
from cortical_mirror.scene.front import NormalisedFront
from cortical_mirror.scene.graph import SceneGraph
from cortical_mirror.scene.video import VideoGraph
from cortical_mirror.scene.world import SpriteWorld


def panning(frames: int, seed: int = 0):
    """A camera panning right at 2 px per frame over a still world with two movers."""

    world = SpriteWorld(seed=seed, width=640, height=192, objects=2, speed=(0.5, 1.0))
    for t in range(frames):
        world.step()
        yield look(world.render(), (128 + 2 * t, 96), (256, 192))


def errors(graph, frames):
    expected, repeated, previous = [], [], None
    for t, frame in enumerate(frames):
        if previous is not None and t > 40:
            expected.append(float(np.mean((graph.expect() - frame) ** 2)))
            repeated.append(float(np.mean((previous - frame) ** 2)))
        graph.observe(frame)
        previous = frame
    return np.mean(expected), np.mean(repeated)


def test_registering_the_camera_beats_repeating_the_last_frame_while_it_pans():
    expected, repeated = errors(VideoGraph(), panning(120))
    assert expected < 0.8 * repeated


def test_without_registration_a_pan_is_not_predicted():
    graph = SceneGraph()
    expected, repeated = errors(_Plain(graph), panning(120))
    assert expected > repeated


def test_the_video_graph_can_match_in_the_normalised_code():
    graph = VideoGraph(front=NormalisedFront())
    expected, repeated = errors(graph, panning(120))
    assert expected < repeated
    assert graph.to_json()["camera"]["moved"][0] > 100


class _Plain:
    """v0.4's graph on video: its expectation, no registration."""

    def __init__(self, graph):
        self.graph = graph

    def observe(self, frame):
        self.graph.observe(frame)

    def expect(self):
        return self.graph.expect(shade=True)
