import numpy as np

from cortical_mirror.engines import CopyEngine, EnginePrediction
from cortical_mirror.ensemble import AdaptiveEngine


class BlackEngine:
    def __init__(self, width, height):
        self.width, self.height = width, height

    def reset(self):
        return None

    def predict(self, frame):
        return EnginePrediction(
            image=np.zeros_like(frame),
            confidence=np.zeros(frame.shape[:2], dtype=np.float32),
            observed_error=None,
            inference_ms=0.0,
        )


def test_routing_learns_which_predictor_works_from_due_errors():
    engine = AdaptiveEngine(
        {"copy": CopyEngine(64, 48), "black": BlackEngine(64, 48)},
        temperature=0.01,
        adaptation=0.5,
        patch_size=8,
    )
    rng = np.random.default_rng(8)
    base = rng.integers(40, 220, (48, 64, 3), dtype=np.int16)
    for _ in range(6):
        frame = np.clip(base + rng.integers(-2, 3, base.shape), 0, 255).astype(np.uint8)
        result = engine.predict(frame)
    assert engine.last_weights[..., 0].mean() > 0.99
    assert np.mean(result.image) > 40
    assert result.observed_error is not None
    assert result.observed_error.mean() < 0.02


def test_first_forecast_has_no_error_history():
    engine = AdaptiveEngine({"copy": CopyEngine(64, 48)})
    frame = np.full((48, 64, 3), 127, dtype=np.uint8)
    result = engine.predict(frame)
    assert result.observed_error is None
    assert np.array_equal(result.image, frame)
    assert np.allclose(engine.last_weights, 1.0)
