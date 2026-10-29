"""The efference copy, learned: which motion cells each eye command makes active.

For every command the map holds the mean responsibility of each MT motion type on
the frames right after it, learned by a local delta rule (Hebbian averaging). A
newborn's map is uniform: no command predicts anything. The motion a command is
expected to cause is read from the same learned cells (their displacements),
and whatever the cells sense beyond it is the world moving.
"""

from __future__ import annotations

import torch


class Efference:
    def __init__(self, commands: int = 9, types: int = 16, rate: float = 0.05, device="cpu"):
        self.rate = rate
        self.map = torch.full((commands, types), 1.0 / types, device=device)

    def learn(self, command: int, responsibilities: torch.Tensor) -> None:
        """Move the command's expectation toward the motion types just active."""

        seen = responsibilities.mean((0, 2, 3)).to(self.map.device)
        self.map[command] += self.rate * (seen - self.map[command])

    def self_velocity(self, command: int, displacements: torch.Tensor) -> torch.Tensor:
        """``[2]``: the motion the command is expected to cause: what is sensed after it,
        minus what is sensed after no command (the world's own motion on average).
        Zero for "no command", and for every command of a newborn's uniform map."""

        difference = self.map[command] - self.map[0]
        return difference @ displacements.to(self.map.device)

    def world_motion(
        self, velocity: torch.Tensor, command: int, displacements: torch.Tensor
    ) -> torch.Tensor:
        """``[B, 2, H, W]``: sensed velocity minus what the command explains."""

        return velocity - self.self_velocity(command, displacements)[None, :, None, None]

    def state_dict(self) -> dict:
        return {"map": self.map.cpu(), "rate": self.rate}

    @classmethod
    def from_state(cls, state: dict, device="cpu") -> Efference:
        commands, types = state["map"].shape
        efference = cls(commands, types, state["rate"], device)
        efference.map = state["map"].to(device)
        return efference
