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
