import copy

import torch

from cortical_mirror.model import CorticalFlowNet, warp_forward


def small_model() -> CorticalFlowNet:
    return CorticalFlowNet(channels=(4, 8, 12), max_displacement=4.0)


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
        for head in (first.flow_head, first.residual_head):
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
    for index in range(5):
        output = model(torch.rand(1, 3, 16, 24), state)
        if output.has_ground_truth:
            loss = loss + model.loss(output)
        state = output.state
    loss.backward()
    gradient = model.flow_head.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
    confidence_gradient = model.confidence_head.weight.grad
    assert confidence_gradient is not None
    assert torch.isfinite(confidence_gradient).all()
    assert confidence_gradient.abs().sum() > 0
