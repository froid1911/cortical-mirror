import torch

from cortical_mirror.dmvfn_impl import DMVFN


def test_dmvfn_is_deterministic_and_device_safe_on_cpu():
    torch.manual_seed(3)
    model = DMVFN().eval()
    frames = torch.rand(1, 6, 64, 64)
    with torch.no_grad():
        first = model(frames, training=False, deterministic=True)
        second = model(frames, training=False, deterministic=True)
    assert first
    assert first[-1].shape == (1, 3, 64, 64)
    assert torch.isfinite(first[-1]).all()
    assert torch.equal(first[-1], second[-1])
