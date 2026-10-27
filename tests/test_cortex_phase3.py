import numpy as np
import torch

from cortical_mirror.cortex.engine import CortexEngine
from cortical_mirror.cortex.eye import Eye


def moving_world(frames: int) -> list[np.ndarray]:
    """A still textured scene with one block moving back and forth in its right half."""

    rng = np.random.default_rng(0)
    base = rng.integers(60, 200, (192, 256), dtype=np.uint8)
    base = np.ascontiguousarray(np.repeat(np.repeat(base[::4, ::4], 4, 0), 4, 1))
    out = []
    for t in range(frames):
        frame = base.copy()
        x = 170 + int(40 * np.sin(t / 6))
        frame[80:112, x : x + 32] = 255
        out.append(np.repeat(frame[..., None], 3, axis=2))
    return out


def test_eye_decide_moves_gaze_for_the_next_crop():
    eye = Eye(period_frames=1)
    surprise = torch.zeros(1, 1, 48, 64)
    surprise[..., 24, 63] = 1.0
    world = torch.rand(1, 1, 192, 256)
    before = eye.crop(world)
    command = eye.decide(surprise)
    assert command == 1
    assert not torch.equal(eye.crop(world), before)


def test_periphery_error_map_marks_moving_region():
    # Averaged over frames: the block moves sinusoidally, so it is nearly still at its
    # turning points, and a newborn's own drifting representation adds noise everywhere.
    engine = CortexEngine(phase=3)
    moving = still = 0.0
    for index, frame in enumerate(moving_world(120)):
        engine.predict(frame)
        if index >= 40:
            surprise = engine.last_surprise[0, 0]
            moving += float(surprise[:, 36:].mean())
            still += float(surprise[:, :28].mean())
    assert moving > 1.5 * still


def test_phase3_engine_shows_the_world_and_moves_its_eye():
    engine = CortexEngine(phase=3)
    commands = []
    for frame in moving_world(60):
        result = engine.predict(frame)
        commands.append(engine.last_command)
    assert result.image.shape == (192, 256, 3)
    assert any(commands)
    assert len(engine.eyes[0].history) > 1


def test_world_view_marks_where_the_eye_has_been():
    from cortical_mirror.cortex.engine import render_world

    code = torch.zeros(1, 4, 96, 128)
    frame = np.full((192, 256, 3), 128, np.uint8)
    image = render_world(code, frame, (136, 96), history=[(128, 96), (136, 96)])
    blue, green, red = (int(v) for v in image[96, 128])
    assert blue > red + 40  # a fixation dot in the expectation colour
