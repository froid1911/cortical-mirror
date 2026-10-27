import cv2
import numpy as np
import torch

from cortical_mirror.cortex.engine import CortexEngine
from cortical_mirror.cortex.twin import TwinEngine
from cortical_mirror.server import LiveState
from tests.cortex_fixtures import pan


def frames(count=8):
    out = []
    for frame in pan(count, 2, 96, 128):
        gray = (frame[0, 0].numpy() * 255).astype(np.uint8)
        out.append(np.repeat(gray[..., None], 3, axis=2))
    return out


def test_twin_reset_affects_newborn_only():
    twin = TwinEngine(CortexEngine(seed=1), CortexEngine(seed=2))
    for frame in frames(6):
        twin.predict(frame)
    adult_weights = twin.adult.hierarchy.levels[0].W.clone()
    twin.request_newborn_reset()  # as the browser's thread does
    assert twin.newborn.age_frames == 6
    twin.predict(frames(1)[0])
    assert twin.newborn.age_frames == 1
    assert twin.adult.age_frames == 7
    assert not torch.equal(
        twin.adult.hierarchy.levels[0].W, CortexEngine(seed=2).hierarchy.levels[0].W
    )
    assert adult_weights.shape == twin.adult.hierarchy.levels[0].W.shape


def test_server_publishes_extra_views(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 25, (128, 96))
    for frame in frames(6):
        writer.write(frame)
    writer.release()
    state = LiveState(TwinEngine(CortexEngine(), CortexEngine()), str(path), "twins", False, 5)
    state.run()
    assert not state.error
    for view in ("newborn", "adult", "newborn_cells", "adult_cells"):
        assert state.latest(view)[1].startswith(b"\xff\xd8")
    metrics = state.metrics()
    assert metrics["mode"] == "twins"
    assert metrics["twins"]["newborn"]["age_frames"] == 5


def test_dreaming_twins_learn_nothing(tmp_path):
    twin = TwinEngine(CortexEngine(seed=1), CortexEngine(seed=2))
    for frame in frames(4):
        twin.predict(frame)
    weights = [e.hierarchy.levels[0].W.clone() for e in (twin.newborn, twin.adult)]
    ages = (twin.newborn.age_frames, twin.adult.age_frames)
    state = LiveState(twin, "0", "twins", False, 0)
    state.control({"eyes_closed": True})  # as the space key does
    for frame in frames(4):
        twin.predict(frame)
    assert (twin.newborn.age_frames, twin.adult.age_frames) == ages
    assert all(
        torch.equal(e.hierarchy.levels[0].W, w)
        for e, w in zip((twin.newborn, twin.adult), weights, strict=True)
    )
    state.control({"eyes_closed": False})
    twin.predict(frames(1)[0])
    assert twin.newborn.age_frames == ages[0] + 1
