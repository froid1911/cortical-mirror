import numpy as np
import torch

from cortical_mirror.cortex.engine import CortexEngine
from tests.cortex_fixtures import pan


def as_bgr(frame: torch.Tensor) -> np.ndarray:
    gray = (frame[0, 0].numpy() * 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


FRAMES = [as_bgr(f) for f in pan(30, 2, 96, 128)]


def test_engine_predicts_a_frame_of_the_mirror_size():
    engine = CortexEngine()
    for frame in FRAMES[:3]:
        result = engine.predict(frame)
    assert result.image.shape == (192, 256, 3) and result.image.dtype == np.uint8
    assert engine.receptive_fields().ndim == 3


def test_learning_engine_ages_and_frozen_engine_does_not():
    learning, frozen = CortexEngine(learn=True), CortexEngine(learn=False)
    for frame in FRAMES[:10]:
        learning.predict(frame)
        frozen.predict(frame)
    assert learning.age_frames == 10
    assert frozen.age_frames == 0
    assert np.isfinite(learning.last_code_error) and np.isfinite(learning.last_persistence_error)


def test_reset_learning_restores_newborn_weights():
    engine = CortexEngine()
    newborn = engine.hierarchy.levels[0].W.clone()
    for frame in FRAMES:
        engine.predict(frame)
    assert not torch.equal(engine.hierarchy.levels[0].W, newborn)
    engine.reset_learning()
    assert torch.equal(engine.hierarchy.levels[0].W, newborn)
    assert engine.age_frames == 0


def test_save_and_load_round_trip(tmp_path):
    engine = CortexEngine()
    for frame in FRAMES:
        engine.predict(frame)
    engine.save(tmp_path / "cortex.pt")
    loaded = CortexEngine.load(tmp_path / "cortex.pt", learn=False)
    assert loaded.age_frames == engine.age_frames
    assert torch.equal(loaded.hierarchy.levels[0].W, engine.hierarchy.levels[0].W)
    assert not loaded.learn


def test_experience_counts_every_stream_and_survives_saving(tmp_path):
    engine = CortexEngine()
    engine.streams = 16
    for frame in FRAMES[:5]:
        engine.predict(frame)
    assert engine.experience_frames == 80
    engine.save(tmp_path / "adult.pt")
    assert CortexEngine.load(tmp_path / "adult.pt").experience_frames == 80
