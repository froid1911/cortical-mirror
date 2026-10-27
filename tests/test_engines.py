from types import SimpleNamespace

import numpy as np
import torch

from cortical_mirror.engines import CorticalEngine
from cortical_mirror.model import CorticalFlowNet
from cortical_mirror.train import save_checkpoint


def test_checkpoint_round_trip_supports_path_metadata(tmp_path):
    model = CorticalFlowNet(channels=(4, 8, 12), strides=(1, 2, 4))
    optimizer = torch.optim.AdamW(model.parameters())
    args = SimpleNamespace(width=64, height=48, data=tmp_path / "videos")
    save_checkpoint(tmp_path / "model.pt", model, optimizer, 0, {"loss": 1.0}, args)

    engine = CorticalEngine.from_checkpoint(tmp_path / "model.pt", device="cpu")
    frame = np.random.default_rng(4).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    first = engine.predict(frame)
    second = engine.predict(frame)
    assert first.image.shape == frame.shape
    assert np.array_equal(first.image, frame)
    assert second.observed_error is not None


def test_cortex_view_shows_every_level_expectation_over_surprise():
    from cortical_mirror.server import cortex_view

    engine = CorticalEngine(
        CorticalFlowNet(channels=(4, 8, 12), strides=(1, 2, 4)), device="cpu", width=64, height=48
    )
    frame = np.random.default_rng(1).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    engine.predict(frame)
    engine.predict(frame)
    assert [expected.shape[:2] for expected, _ in engine.last_levels] == [
        (48, 64),
        (24, 32),
        (12, 16),
    ]
    view = cortex_view(engine, 64, 48)
    assert view.shape == (96, 192, 3)
    # An untrained network repeats the frame, so level 1 (left) expects exactly what it
    # saw, shown mirrored.
    assert np.abs(view[:48, :64].astype(int) - frame[:, ::-1]).max() <= 1
