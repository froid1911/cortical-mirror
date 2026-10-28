import copy
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np

from cortical_mirror.scene.live import Handler, SceneLive, SpriteSource


def run(state: SceneLive, frames: int) -> None:
    for _ in range(frames):
        state.step()


def test_the_sprite_source_gives_gray_frames_and_loops_forever():
    source = SpriteSource(seed=0)
    frames = [source.read() for _ in range(3)]
    assert all(f.shape == (192, 256) and f.dtype == np.float32 for f in frames)


def test_each_step_publishes_what_is_seen_what_is_explained_and_the_graph():
    state = SceneLive(SpriteSource(seed=1))
    run(state, 60)
    for view in ("seen", "explained"):
        version, jpeg = state.latest(view)
        assert version > 0 and jpeg.startswith(b"\xff\xd8")
    graph = state.scene()
    assert graph["nodes"] and {"id", "x", "y", "size", "state"} <= graph["nodes"][0].keys()
    assert graph["edits"] == {"deleted": [], "moved": {}, "duplicated": []}


def test_a_deleted_node_stays_deleted_in_the_explained_view_while_the_graph_keeps_it():
    state = SceneLive(SpriteSource(seed=1))
    run(state, 60)
    node = state.scene()["nodes"][0]["id"]
    state.control({"delete": node})
    run(state, 5)
    assert node in [n["id"] for n in state.scene()["nodes"]]
    assert state.scene()["edits"]["deleted"] == [node]
    assert not np.allclose(state.explained, state.graph.render())


def test_moving_and_duplicating_are_remembered_and_reset_clears_them():
    state = SceneLive(SpriteSource(seed=1))
    run(state, 60)
    node = state.scene()["nodes"][0]["id"]
    state.control({"move": node, "by": [20, -10]})
    state.control({"move": node, "by": [5, 0]})
    state.control({"duplicate": node})
    edits = state.scene()["edits"]
    assert edits["moved"] == {str(node): [25.0, -10.0]} and edits["duplicated"] == [node]
    state.control({"reset": True})
    assert state.scene()["edits"] == {"deleted": [], "moved": {}, "duplicated": []}


def test_the_page_and_the_graph_are_served_over_http():
    state = SceneLive(SpriteSource(seed=1))
    run(state, 30)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.state = state
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "scene graph" in page.lower()
        graph = json.loads(urllib.request.urlopen(base + "/scene.json").read())
        assert "nodes" in graph and "edges" in graph
        request = urllib.request.Request(
            base + "/edit", data=json.dumps({"reset": True}).encode(), method="POST"
        )
        assert json.loads(urllib.request.urlopen(request).read())["ok"]
    finally:
        server.shutdown()
        server.server_close()


def test_a_fragment_on_a_deleted_object_is_deleted_with_it():
    state = SceneLive(SpriteSource(seed=1))
    run(state, 60)
    node = state.graph.nodes[0]
    twin = copy.deepcopy(node)
    twin.id = 999
    state.graph.nodes.append(twin)
    state.control({"delete": node.id})
    state.graph.observe = lambda image: None  # hold the graph still for this check
    state.step()
    hidden = {n.id for n in state.graph.nodes if n.state != "seen"}
    alone = state.graph.edit(delete={node.id, 999} | hidden)
    assert np.allclose(state.explained, alone)


def test_impossible_edits_are_refused_and_change_nothing():
    import pytest

    state = SceneLive(SpriteSource(seed=1))
    run(state, 30)
    for bad in (
        {"move": 0, "by": [float("inf"), 0]},
        {"move": 0, "by": [1e999, 0]},
        {"delete": 1e999},
        {"delete": 1, "move": 2, "by": "x"},
        {"move": 0, "by": [1, 2, 3]},
    ):
        with pytest.raises(ValueError):
            state.control(bad)
    assert state.scene()["edits"] == {"deleted": [], "moved": {}, "duplicated": []}


def test_a_move_is_kept_within_reach():
    state = SceneLive(SpriteSource(seed=1))
    for _ in range(10):
        state.control({"move": 0, "by": [1e300, -1e300]})
    dx, dy = state.scene()["edits"]["moved"]["0"]
    assert abs(dx) <= 512 and abs(dy) <= 384


def test_a_duplicate_follows_its_original_and_goes_when_it_is_deleted():
    state = SceneLive(SpriteSource(seed=1))
    run(state, 60)
    node = next(n for n in state.graph.nodes if n.state == "seen")
    state.control({"duplicate": node.id})
    state.control({"delete": node.id})
    state.graph.observe = lambda image: None
    state.step()
    hidden = {n.id for n in state.graph.nodes if n.state != "seen"}
    assert np.allclose(state.explained, state.graph.edit(delete={node.id} | hidden))


def test_a_new_sprite_scene_starts_a_new_graph_and_clears_the_edits():
    state = SceneLive(SpriteSource(seed=1, scene_frames=40))
    run(state, 39)
    state.control({"delete": 0})
    run(state, 2)
    assert state.graph.frame_count <= 2
    assert state.scene()["edits"]["deleted"] == []


def test_the_server_answers_bad_requests_with_400():
    state = SceneLive(SpriteSource(seed=1))
    run(state, 5)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.state = state
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        for body in (b'{"delete": 1e999}', b"[1]", b"not json"):
            request = urllib.request.Request(base + "/edit", data=body, method="POST")
            try:
                urllib.request.urlopen(request)
                raise AssertionError("expected 400")
            except urllib.error.HTTPError as error:
                assert error.code == 400
    finally:
        server.shutdown()
        server.server_close()


def test_a_whole_group_can_be_deleted_at_once():
    state = SceneLive(SpriteSource(seed=1))
    run(state, 30)
    state.control({"delete_all": [0, 1, 2]})
    assert state.scene()["edits"]["deleted"] == [0, 1, 2]
    import pytest

    with pytest.raises(ValueError):
        state.control({"delete_all": [0, "x"]})
    with pytest.raises(ValueError):
        state.control({"delete_all": list(range(1000))})


def test_video_sources_run_through_the_camera_registering_graph():
    from cortical_mirror.scene.video import VideoGraph

    state = SceneLive(SpriteSource(seed=1), video=VideoGraph())
    run(state, 40)
    scene = state.scene()
    assert "camera" in scene and scene["nodes"]
    assert state.graph is state.video.graph
