"""Online learning: the hierarchy keeps fitting the camera stream while it predicts.

Inference never waits for learning. A background thread trains a copy of the
network on short windows replayed from the last few seconds of video, using the
same rule as offline training: a prediction is scored only against the frame it
was issued for. Improved weights are copied into the live network between frames.
"""

from __future__ import annotations

import copy
import random
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .model import CorticalFlowNet


class RatioMeter:
    """Smoothed prediction error relative to persistence error.

    Errors are averaged before dividing: when the camera is almost still the
    per-frame ratio explodes on errors that are negligible in absolute terms.
    """

    def __init__(self, decay: float, warmup: int = 20):
        self.decay = decay
        self.warmup = warmup
        self.reset()

    def reset(self) -> None:
        self.prediction = self.copied = 0.0
        self.count = 0

    def update(self, prediction: float, copied: float) -> None:
        if not np.isfinite(prediction):
            prediction = 10.0 * copied
        self.prediction = self.decay * self.prediction + (1 - self.decay) * prediction
        self.copied = self.decay * self.copied + (1 - self.decay) * copied
        self.count += 1

    @property
    def ratio(self) -> float:
        if self.count < self.warmup or self.copied <= 0:
            return 1.0
        return self.prediction / self.copied


@dataclass
class LearnerStats:
    steps: int
    frames_seen: int
    loss: float
    prediction_mae: float
    copy_mae: float
    paused: bool
    rollbacks: int
    learning_rate: float
    error: str


class OnlineLearner:
    def __init__(
        self,
        model: CorticalFlowNet,
        learning_rate: float = 2e-4,
        window: int = 6,
        batch_size: int = 2,
        memory: int = 300,
        steps_per_frame: float = 1.0,
        anchor_weight: float = 1e-2,
        recent: int = 30,
        burn_in: int = 8,
        duty_cycle: float = 0.9,
    ):
        if window < 3:
            raise ValueError("window must be at least 3 frames")
        self.live = model
        self.student = copy.deepcopy(model).train()
        for parameter in self.student.parameters():
            parameter.requires_grad_(True)
        self.initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
        self.learning_rate = learning_rate
        self.window = window
        self.batch_size = batch_size
        self.steps_per_frame = steps_per_frame
        self.anchor_weight = anchor_weight
        self.recent = recent
        self.burn_in = burn_in
        # Fraction of wall time the learner may spend training; the rest is left to
        # inference so learning never makes the mirror stutter.
        self.duty_cycle = duty_cycle
        device = next(model.parameters()).device
        # A separate stream: inference waits only for its own kernels, never for
        # queued training work. Measured on an RTX 3090 (batch 2, burn-in 8): about
        # 15 steps/s while inference stays near 18 ms per frame.
        self._stream = torch.cuda.Stream(device, priority=0) if device.type == "cuda" else None
        # Called after a rollback so the live network can drop state built by bad weights.
        self.on_rollback = None
        self.frames: deque[torch.Tensor] = deque(maxlen=memory)
        # `lock` guards the live weights against inference; the others guard the
        # student network and the replay buffer.
        self.lock = threading.Lock()
        self._train_lock = threading.Lock()
        self._buffer_lock = threading.Lock()
        self.paused = False
        self.error = ""
        self._optimizer = self._make_optimizer()
        self._running = False
        self._thread: threading.Thread | None = None
        self._steps = 0
        self._frames_seen = 0
        self._loss = self._prediction = self._copy = float("nan")
        self._reset_guard()

    def _reset_guard(self) -> None:
        # The starting weights are the reference: persistence-level or better.
        self._replay = RatioMeter(decay=0.9)
        self._live = RatioMeter(decay=0.95)
        self._best_ratio = 1.0
        self._good = {key: value.clone() for key, value in self.initial.items()}
        self._rollbacks = 0
        for group in self._optimizer.param_groups:
            group["lr"] = self.learning_rate

    def _guard(self, prediction: float, copied: float) -> None:
        """Roll back to the last good weights if learning starts to diverge.

        The ratio of replayed prediction error to persistence error is smoothed;
        a fresh network legitimately rises a little above 1 before improving, so
        only a large excursion counts as divergence.
        """

        self._replay.update(prediction, copied)
        # Replay windows can look fine while the long-running live network drifts,
        # so the worse of the two decides.
        worst = max(self._replay.ratio, self._live.ratio)
        if worst < self._best_ratio and self._steps % 10 == 0:
            self._best_ratio = worst
            self._good = {k: v.detach().clone() for k, v in self.student.state_dict().items()}
        diverged = worst > max(2.0, 1.5 * self._best_ratio)
        if diverged or not np.isfinite(prediction):
            self.student.load_state_dict(self._good)
            rate = max(self._optimizer.param_groups[0]["lr"] * 0.5, self.learning_rate / 4)
            self._optimizer = self._make_optimizer()
            for group in self._optimizer.param_groups:
                group["lr"] = rate
            self._replay.reset()
            self._live.reset()
            self._rollbacks += 1
            if self.on_rollback is not None:
                self.on_rollback()

    def _make_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.student.parameters(), lr=self.learning_rate)

    def observe(self, frame: torch.Tensor) -> None:
        """Remember one camera frame, ``[1, 3, H, W]`` in ``[0, 1]``."""

        if self.paused:
            return
        with self._buffer_lock:
            self.frames.append(frame.detach()[0].to(torch.float16))
            self._frames_seen += 1

    def observe_outcome(self, prediction_mae: float, copy_mae: float) -> None:
        """Record how the live network's latest due prediction compared with persistence."""

        if not self.paused:
            self._live.update(prediction_mae, copy_mae)

    def _sample(self) -> tuple[torch.Tensor, int]:
        with self._buffer_lock:
            frames = list(self.frames)
        burn_in = max(0, min(self.burn_in, len(frames) - self.window))
        span = burn_in + self.window
        latest_start = len(frames) - span
        recent_start = max(0, latest_start - self.recent)
        starts = [
            # Half the windows come from the last second: adapt to what is happening now.
            random.randint(recent_start, latest_start)
            if index % 2 == 0
            else random.randint(0, latest_start)
            for index in range(self.batch_size)
        ]
        clips = [torch.stack(frames[start : start + span]) for start in starts]
        return torch.stack(clips).float(), burn_in

    def step(self) -> float:
        """One gradient step on replayed windows; returns the loss."""

        with self._train_lock:
            return self._step()

    def _step(self) -> float:
        clips, burn_in = self._sample()
        state = None
        if burn_in:
            # Train from a settled recurrent state, as the live network runs.
            with torch.no_grad():
                for index in range(burn_in):
                    state = self.student(clips[:, index], state).state.detached()
        losses, predicted, copied = [], [], []
        for index in range(burn_in, clips.shape[1]):
            output = self.student(clips[:, index], state)
            if output.has_ground_truth:
                losses.append(self.student.loss(output))
                predicted.append(output.observed_error.mean())
                copied.append((clips[:, index] - clips[:, index - 1]).abs().mean())
            state = output.state
        loss = torch.stack(losses).mean()
        if self.anchor_weight:
            # Elastic pull toward the starting weights keeps a pretrained network from
            # forgetting everything when the camera points at a wall for a minute.
            drift = sum(
                (parameter - self.initial[name]).square().sum()
                for name, parameter in self.student.named_parameters()
            )
            loss = loss + self.anchor_weight * drift
        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if torch.isfinite(clip_grad_norm_(self.student.parameters(), 1.0)):
            self._optimizer.step()
        self._steps += 1
        self._loss = float(loss.detach())
        self._prediction = float(torch.stack(predicted).mean())
        self._copy = float(torch.stack(copied).mean())
        self._guard(self._prediction, self._copy)
        self.publish()
        return self._loss

    def publish(self) -> None:
        with self.lock, torch.no_grad():
            for live, learned in zip(
                self.live.parameters(), self.student.parameters(), strict=True
            ):
                live.copy_(learned)
            if self._stream is not None:
                # Inference runs on another stream: finish the copy before it reads.
                self._stream.synchronize()

    def reset(self) -> None:
        """Forget everything learned live and return to the starting weights."""

        with self._train_lock, self.lock:
            self.student.load_state_dict(self.initial)
            self.live.load_state_dict(self.initial)
            self._optimizer = self._make_optimizer()
            self._steps = 0
            self._frames_seen = 0
            self._loss = self._prediction = self._copy = float("nan")
            self.error = ""
            self._reset_guard()

    def ready(self) -> bool:
        if self.paused or len(self.frames) < self.window:
            return False
        return self._steps < self._frames_seen * self.steps_per_frame

    def _run(self) -> None:
        while self._running:
            if not self.ready():
                time.sleep(0.002)
                continue
            started = time.perf_counter()
            try:
                if self._stream is not None:
                    with torch.cuda.stream(self._stream):
                        self.step()
                else:
                    self.step()
            except Exception as error:  # keep the mirror alive; show why learning stopped
                self.error = f"{type(error).__name__}: {error}"
                self.paused = True
            if 0 < self.duty_cycle < 1:
                elapsed = time.perf_counter() - started
                time.sleep(elapsed * (1 - self.duty_cycle) / self.duty_cycle)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="online-learner", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def stats(self) -> LearnerStats:
        return LearnerStats(
            steps=self._steps,
            frames_seen=self._frames_seen,
            loss=self._loss,
            prediction_mae=self._prediction,
            copy_mae=self._copy,
            paused=self.paused,
            rollbacks=self._rollbacks,
            learning_rate=self._optimizer.param_groups[0]["lr"],
            error=self.error,
        )
