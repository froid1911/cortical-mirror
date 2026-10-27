import numpy as np
import torch

from cortical_mirror.cortex.retina import Retina, frame_to_gray


def spot(value: float, background: float = 0.5) -> torch.Tensor:
    image = torch.full((1, 1, 96, 128), background)
    image[..., 46:50, 62:66] = value
    return image


def test_retina_uniform_frame_is_silent():
    for level in (0.0, 0.5, 1.0):
        code = Retina().encode(torch.full((1, 1, 96, 128), level))
        assert code.shape == (1, 2, 96, 128)
        assert torch.isfinite(code).all()
        assert code.abs().max() < 1e-6


def test_bright_spot_drives_on_centre():
    code = Retina().encode(spot(1.0))
    assert code[0, 0, 48, 64] > 0.5
    assert code[0, 1, 48, 64] == 0


def test_dark_spot_drives_off_centre():
    code = Retina().encode(spot(0.0))
    assert code[0, 1, 48, 64] > 0.5
    assert code[0, 0, 48, 64] == 0


def test_lagged_channels_repeat_previous_frame():
    retina = Retina(lagged=True)
    first, second = spot(1.0), spot(0.0)
    a = retina.encode(first)
    b = retina.encode(second)
    assert a.shape == (1, 4, 96, 128) and retina.channels == 4
    assert torch.equal(a[:, 2:], torch.zeros_like(a[:, 2:]))
    assert torch.allclose(b[:, 2:], a[:, :2])
    retina.reset()
    assert torch.equal(retina.encode(first)[:, 2:], torch.zeros_like(a[:, 2:]))


def test_retina_accepts_any_frame_size():
    rng = np.random.default_rng(0)
    for frame in (
        rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
        rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8),
        rng.integers(0, 256, (240, 320), dtype=np.uint8),
    ):
        gray = frame_to_gray(frame, 128, 96, "cpu")
        assert gray.shape == (1, 1, 96, 128)
        assert 0.0 <= float(gray.min()) <= float(gray.max()) <= 1.0
        assert Retina().encode(gray).shape == (1, 2, 96, 128)
