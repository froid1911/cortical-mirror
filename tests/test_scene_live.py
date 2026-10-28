import copy
import json
import threading
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
