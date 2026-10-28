"""Where to look, learned: the value of a fixation is the learning it brings.

Every few frames the eye picks one of the saccade commands. Each possible target
is described by a few features of what the scene graph knows about the place the
fovea would land on:

* how unknown its background is;
* how badly the graph has been predicting it;
* how new the objects there are;
* how long the fovea has already spent there;
A target's value is a weighted sum of its features, plus the place's own learned
value: the progress looking there brought before, a map that starts optimistic
and is raised again where something new appears. After the fixation, the
weights change by a three-factor rule: the features of the fixation just made
(presynaptic) times a global signal, the learning progress it brought (how much
the graph's prediction error in the fovea fell) minus the progress the weights
had promised, like a dopamine prediction error. A newborn's weights are zero,
so it looks at random.

``policy="surprise"`` replaces the learned value by the prediction error alone:
the reflex the v0.3 eye had built in.
"""

from __future__ import annotations

import random

import numpy as np

from ..cortex.eye import COMMANDS

FEATURES = ("unknown", "error", "novelty", "familiar", "bias")
# The prediction error (mean squared, luminance) that counts as fully surprising.
ERROR_SCALE = 0.02
# Frames in the fovea a node takes to lose most of its novelty, and a place to
# count as fully familiar.
FAMILIAR = 30.0
# Frames a node must have lasted to count as a thing that is new.
LASTING = 10


class Attention:
    def __init__(
        self,
        weights=None,
        learn: bool = False,
        policy: str = "learned",
        period: int = 10,
        rate: float = 0.2,
        explore: float = 0.0,
        seed: int = 0,
    ):
        self.weights = np.zeros(len(FEATURES)) if weights is None else np.array(weights, float)
        self.learn_weights = learn
        self.policy = policy
        self.period = period
        self.rate = rate
        self.explore = explore
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        self.history: list[dict] = []
        self._fixation: dict | None = None
        self._recent: list[tuple[float, float]] = []

    def _target(self, graph, command: int) -> tuple[float, float]:
        planned = graph.motor[command]
        return graph.gaze[0] + planned[0], graph.gaze[1] + planned[1]

    def features(self, graph) -> np.ndarray:
        """``[commands, features]``: what the graph knows about each possible target."""

        out = np.zeros((len(COMMANDS), len(FEATURES)))
        half = graph.fovea_size // 2
        positions = graph.world_positions()
        # Only things that last count as new: noise throws up nodes that live a frame.
        looked = {node.id: node.looked for node in graph.inner.nodes if node.age >= LASTING}
        for command in range(len(COMMANDS)):
            x, y = (int(round(v)) for v in self._target(graph, command))
            area = (slice(y - half, y + half), slice(x - half, x + half))
            novelty = sum(
                np.exp(-looked.get(node, 0) / FAMILIAR)
                for node, nx, ny in positions
                if node in looked and abs(nx - x) < half and abs(ny - y) < half
            )
            out[command] = (
                float(np.mean(1.0 - graph.confidence[area])),
                min(1.0, float(np.mean(graph.error[area])) / ERROR_SCALE),
                min(1.0, novelty),
                min(1.0, float(np.mean(graph.visits[area])) / FAMILIAR),
                1.0,
            )
        return out

    def choose(self, graph, allowed) -> int:
        allowed = list(allowed)
        if self.explore and self.rng.random() < self.explore:
            return self.rng.choice(allowed)
        features = self.features(graph)
        if self.policy == "surprise":
            values = features[:, FEATURES.index("error")].copy()
        elif self.policy == "random":
            values = np.zeros(len(COMMANDS))
        else:
            # The learned value of a target: its features, weighted, plus what looking
            # at that very place has taught before (its own learned value).
            values = features @ self.weights + self.place_values(graph)
        # Inhibition of return: places just looked at are worth less.
        for command in allowed:
            tx, ty = self._target(graph, command)
            if any(abs(tx - x) < 8 and abs(ty - y) < 8 for x, y in self._recent):
                values[command] -= 1.0
        best = max(values[c] for c in allowed)
        return self.rng.choice([c for c in allowed if values[c] >= best - 1e-9])

    def place_values(self, graph) -> np.ndarray:
        half = graph.fovea_size // 2
        out = np.zeros(len(COMMANDS))
        for command in range(len(COMMANDS)):
            x, y = (int(round(v)) for v in self._target(graph, command))
            out[command] = float(np.mean(graph.progress[y - half : y + half, x - half : x + half]))
        return out

    def learn(self, features: np.ndarray, progress: float) -> None:
        """Three-factor rule: the fixation's features times (progress − promise)."""

        if self.learn_weights:
            surprise = progress - float(features @ self.weights)
            self.weights += self.rate * surprise * features

    def step(self, graph, t: int, allowed) -> int:
        """The command to issue before frame ``t``: 0 while fixating; at the end of a
        fixation, learn from it and choose the next."""

        if t % self.period:
            return 0
        if self._fixation is not None:
            errors = graph.fovea_errors[self._fixation["start"] :]
            if len(errors) >= 4:
                start, end = np.mean(errors[:2]), np.mean(errors[-2:])
                progress = float((start - end) / (start + 1e-4))
                self._fixation["progress"] = progress
                self.learn(self._fixation["features"], progress)
                graph.remember_progress(self._fixation["target"], progress)
                self.history.append(self._fixation)
        commands = [c for c in range(1, len(COMMANDS)) if allowed(c)] or [0]
        command = self.choose(graph, commands)
        features = self.features(graph)[command]
        self._recent = [*self._recent, self._target(graph, command)][-3:]
        self._fixation = {
            "command": command,
            "target": self._target(graph, command),
            "features": features,
            "start": len(graph.fovea_errors) + 1,
        }
        return command
