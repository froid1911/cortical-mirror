import numpy as np
import torch

from cortical_mirror.engines import CorticalEngine
from cortical_mirror.model import CorticalFlowNet
from cortical_mirror.online import OnlineLearner
from tests.test_model import make_pan


def test_untrained_network_learns_a_pan_from_the_stream_alone():
    torch.manual_seed(0)
    model = CorticalFlowNet(channels=(8, 16, 16), strides=(1, 2, 4), max_displacement=8.0)
    learner = OnlineLearner(
        model, learning_rate=3e-3, window=5, batch_size=4, memory=64, anchor_weight=0.0
    )
    for frame in make_pan(40, 3, seed=1):
        learner.observe(frame[None])
    for _ in range(80):
        learner.step()
    stats = learner.stats()
    assert stats.steps == 80
    assert stats.prediction_mae < 0.8 * stats.copy_mae

    # The live network received the learned weights without being trained itself.
    with torch.no_grad():
        clip = make_pan(6, 3, seed=7)
        state = None
        for frame in clip[:-1]:
            output = model(frame[None], state)
            state = output.state
    assert float(output.flow[0, 0, 4:-4, 4:-8].median()) > 1.5

    learner.reset()
    assert all(
        torch.equal(value, learner.initial[key]) for key, value in model.state_dict().items()
    )


def test_learning_engine_observes_frames_and_never_learns_while_paused():
    engine = CorticalEngine.fresh(
        device="cpu", width=32, height=24, channels=(4, 8, 8), strides=(1, 2, 4)
    )
    learner = OnlineLearner(engine.model, window=3, batch_size=2)
    engine.learner = learner
    frames = (make_pan(4, 2, seed=3).numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    for frame in frames:
        engine.predict(frame)
    assert learner.stats().frames_seen == 4
    learner.paused = True
    engine.predict(frames[0])
    assert learner.stats().frames_seen == 4
    assert not learner.ready()


def test_rollout_does_not_change_the_committed_state():
    torch.manual_seed(2)
    engine = CorticalEngine.fresh(
        device="cpu", width=32, height=24, channels=(4, 8, 8), strides=(1, 2, 4)
    )
    with torch.no_grad():
        for head in engine.model.flow_heads:
            head.weight.normal_(0.0, 0.05)
    frames = (make_pan(3, 2, seed=4).numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    engine.predict(frames[0])
    first = engine.predict(frames[1])
    before = [t.clone() for t in engine.state.predicted]
    imagined = engine.rollout(first.image, 3)
    assert imagined.shape == first.image.shape
    assert all(torch.equal(a, b) for a, b in zip(before, engine.state.predicted, strict=True))


def test_divergence_rolls_back_to_the_last_good_weights_and_slows_down():
    torch.manual_seed(1)
    model = CorticalFlowNet(channels=(4, 8, 8), strides=(1, 2, 4))
    learner = OnlineLearner(model, learning_rate=1e-3, window=3, batch_size=2, anchor_weight=0.0)
    good = {key: value.clone() for key, value in learner.student.state_dict().items()}
    with torch.no_grad():
        for parameter in learner.student.parameters():
            parameter.add_(1.0)
    for _ in range(40):
        learner._guard(prediction=0.08, copied=0.01)
        if learner.stats().rollbacks:
            break
    stats = learner.stats()
    assert stats.rollbacks == 1
    assert stats.learning_rate == 5e-4
    assert all(torch.equal(value, good[key]) for key, value in learner.student.state_dict().items())


def test_live_divergence_alone_triggers_rollback_and_resets_the_engine():
    engine = CorticalEngine.fresh(
        device="cpu", width=32, height=24, channels=(4, 8, 8), strides=(1, 2, 4)
    )
    learner = OnlineLearner(engine.model, window=3, batch_size=2)
    resets = []
    learner.on_rollback = lambda: resets.append(True)
    for _ in range(60):
        learner.observe_outcome(prediction_mae=0.06, copy_mae=0.01)
    learner._guard(prediction=0.009, copied=0.01)  # replay windows look healthy
    assert learner.stats().rollbacks == 1
    assert resets


def test_a_still_camera_does_not_look_like_divergence():
    model = CorticalFlowNet(channels=(4, 8, 8), strides=(1, 2, 4))
    learner = OnlineLearner(model, window=3, batch_size=2)
    for step in range(200):
        # Mostly near-still frames where tiny absolute errors give large ratios.
        still = step % 10
        learner.observe_outcome(0.002 if still else 0.02, 0.0004 if still else 0.03)
        learner._guard(prediction=0.02, copied=0.025)
    assert learner.stats().rollbacks == 0


def test_rollback_reset_is_deferred_to_the_inference_thread():
    engine = CorticalEngine.fresh(
        device="cpu", width=32, height=24, channels=(4, 8, 8), strides=(1, 2, 4)
    )
    engine.learner = OnlineLearner(engine.model, window=3, batch_size=2)
    frames = (make_pan(3, 2, seed=5).numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    engine.predict(frames[0])
    engine.request_reset()  # as the learner thread does after a rollback
    assert engine.state is not None
    result = engine.predict(frames[1])
    assert result.observed_error is None  # the reset happened before this frame


def test_learner_thread_survives_an_exception_and_reports_it():
    import time

    model = CorticalFlowNet(channels=(4, 8, 8), strides=(1, 2, 4))
    learner = OnlineLearner(model, window=3, batch_size=2)

    def broken():
        raise RuntimeError("out of memory")

    learner.step = broken
    for frame in make_pan(4, 2, seed=6):
        learner.observe(frame[None])
    learner.start()
    deadline = time.time() + 5
    while not learner.stats().error and time.time() < deadline:
        time.sleep(0.01)
    learner.stop()
    assert "out of memory" in learner.stats().error
    assert learner.stats().paused
