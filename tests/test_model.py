import copy

import torch
from torch.nn import functional as F

from cortical_mirror.model import CorticalFlowNet, warp_forward


def small_model() -> CorticalFlowNet:
    return CorticalFlowNet(channels=(4, 8, 12), strides=(1, 2, 4), max_displacement=4.0)


def test_zero_history_predicts_persistence():
    model = small_model().eval()
    frame = torch.rand(2, 3, 16, 24)
    with torch.no_grad():
        output = model(frame)
    assert torch.allclose(output.prediction, frame, atol=1e-6)
    assert not output.has_ground_truth
    assert torch.all(output.state.confidence == 0.5)


def test_forward_flow_moves_content_in_flow_direction():
    image = torch.zeros(1, 1, 5, 7)
    image[0, 0, 2, 3] = 1.0
    flow = torch.zeros(1, 2, 5, 7)
    flow[:, 0] = 1.0
    moved = warp_forward(image, flow)
    assert moved[0, 0, 2, 4] > 0.99
    assert moved[0, 0, 2, 3] < 0.01


def test_future_cannot_change_prediction_already_issued():
    torch.manual_seed(4)
    first = small_model().eval()
    with torch.no_grad():
        for head in (*first.flow_heads, first.residual_head):
            head.weight.normal_(0.0, 0.02)
    second = copy.deepcopy(first)
    with torch.no_grad():
        a, b = torch.rand(2, 1, 3, 16, 24)
        future_a = torch.rand(1, 3, 16, 24)
        future_b = torch.rand(1, 3, 16, 24)

        a0 = first(a)
        b0 = second(a)
        a1 = first(b, a0.state)
        b1 = second(b, b0.state)
        assert torch.equal(a1.prediction, b1.prediction)

        a2 = first(future_a, a1.state)
        b2 = second(future_b, b1.state)
        assert not torch.equal(a2.observed_error, b2.observed_error)


def test_observed_next_frame_error_trains_flow_renderer():
    torch.manual_seed(7)
    model = small_model()
    state = None
    loss = 0.0
    for _ in range(5):
        output = model(torch.rand(1, 3, 16, 24), state)
        if output.has_ground_truth:
            loss = loss + model.loss(output)
        state = output.state
    loss.backward()
    gradient = model.flow_heads[0].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
    confidence_gradient = model.confidence_head.weight.grad
    assert confidence_gradient is not None
    assert torch.isfinite(confidence_gradient).all()
    assert confidence_gradient.abs().sum() > 0


def test_upper_level_targets_are_a_fixed_pyramid_and_cannot_collapse():
    model = small_model()
    assert not any("target" in name for name, _ in model.named_parameters())
    frame = torch.rand(1, 3, 16, 24)
    with torch.no_grad():
        first = model(frame)
        second = model(frame, first.state)
    assert torch.allclose(second.state.targets[2], F.avg_pool2d(frame, 4))
    assert len(second.errors) == 3 and second.errors[2].abs().sum() == 0


def make_pan(frames: int, speed: int, seed: int, height: int = 24, width: int = 32):
    """A horizontally panning texture with a natural-image-like 1/f spectrum.

    White noise is avoided on purpose: shifted by more than its correlation length
    it gives photometric gradients no direction, which real camera images do not.
    """

    generator = torch.Generator().manual_seed(seed)
    span = width + speed * frames
    texture = torch.zeros(3, height, span)
    for scale in (1, 2, 4, 8):
        noise = torch.rand(
            1, 3, max(2, height // scale), max(2, span // scale), generator=generator
        )
        texture += scale * F.interpolate(noise, size=(height, span), mode="bilinear")[0]
    texture = (texture - texture.min()) / (texture.max() - texture.min())
    # Content moves right by `speed` pixels per frame.
    return torch.stack(
        [texture[:, :, span - width - speed * t : span - speed * t] for t in range(frames)]
    )


def test_hierarchy_learns_to_predict_a_pan_instead_of_copying():
    torch.manual_seed(0)
    model = CorticalFlowNet(channels=(8, 16, 16), strides=(1, 2, 4), max_displacement=8.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(120):
        clip = make_pan(5, 3, seed=step % 8)
        state, loss = None, 0.0
        for frame in clip:
            output = model(frame[None], state)
            loss = loss + model.loss(output)
            state = output.state
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    clip = make_pan(6, 3, seed=99)
    with torch.no_grad():
        state = None
        for frame in clip[:-1]:
            output = model(frame[None], state)
            state = output.state
    interior = output.flow[0, 0, 4:-4, 4:-8]
    assert float(interior.median()) > 1.5
    model_error = (output.prediction - clip[-1]).abs()[..., 4:-4, 4:-8].mean()
    copy_error = (clip[-2] - clip[-1]).abs()[..., 4:-4, 4:-8].mean()
    assert model_error < 0.6 * copy_error


def test_cell_state_stays_bounded_over_a_long_run():
    torch.manual_seed(3)
    model = CorticalFlowNet(channels=(4, 8, 8), strides=(1, 2, 4), cell_limit=4.0).eval()
    with torch.no_grad():
        for cell in model.cells:
            cell.gates.bias.fill_(3.0)  # forget and input gates wide open: worst case
        state = None
        for _ in range(300):
            output = model(torch.rand(1, 3, 16, 24), state)
            state = output.state
    assert max(float(c.abs().max()) for _, c in state.recurrent) <= 4.0
