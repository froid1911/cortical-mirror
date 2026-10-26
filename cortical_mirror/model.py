"""Hierarchical predictive coding with coarse-to-fine forward-motion rendering.

The design follows the PredNet family (Lotter, Kreiman & Cox, 2017), itself a
deep-learning reading of Rao & Ballard (1999): every level predicts its own next
input and passes only signed prediction errors upward, while recurrent
representations send context downward. Two departures matter here:

* Level targets are a fixed image pyramid, not learned features. A learned target
  can be shrunk to zero to make its own error vanish; a fixed pyramid cannot, so
  upper levels cannot collapse and must predict coarse-scale motion.
* Each level predicts a displacement field, refined coarse to fine. The image is
  rendered by warping what is already seen, so the network must model motion
  instead of hallucinating pixels, and an untrained network is exact persistence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class ModelState:
    """Everything carried from one observed frame to the next."""

    recurrent: list[tuple[Tensor, Tensor]]
    predicted: list[Tensor]
    targets: list[Tensor]
    confidence: Tensor
    flow: Tensor

    def detached(self) -> ModelState:
        return ModelState(
            recurrent=[(h.detach(), c.detach()) for h, c in self.recurrent],
            predicted=[p.detach() for p in self.predicted],
            targets=[t.detach() for t in self.targets],
            confidence=self.confidence.detach(),
            flow=self.flow.detach(),
        )


@dataclass
class StepOutput:
    """Result of observing one frame and issuing a prediction for the next."""

    prediction: Tensor
    state: ModelState
    errors: list[Tensor]
    observed_error: Tensor | None
    previous_confidence: Tensor | None
    previous_flow: Tensor | None

    @property
    def has_ground_truth(self) -> bool:
        return self.observed_error is not None

    @property
    def flow(self) -> Tensor:
        """Displacement, in full-resolution pixels, behind the issued prediction."""

        return self.state.flow


class ConvLSTMCell(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        context_channels: int,
        cell_limit: float = 0.0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.context_channels = context_channels
        self.cell_limit = cell_limit
        self.gates = nn.Conv2d(
            input_channels + hidden_channels + context_channels,
            4 * hidden_channels,
            kernel_size=3,
            padding=1,
        )
        nn.init.xavier_uniform_(self.gates.weight)
        nn.init.zeros_(self.gates.bias)
        with torch.no_grad():
            self.gates.bias[hidden_channels : 2 * hidden_channels].fill_(1.0)

    def forward(
        self,
        x: Tensor,
        state: tuple[Tensor, Tensor] | None,
        context: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        batch, _, height, width = x.shape
        if state is None:
            hidden = x.new_zeros(batch, self.hidden_channels, height, width)
            cell = x.new_zeros(batch, self.hidden_channels, height, width)
        else:
            hidden, cell = state
        if self.context_channels:
            if context is None:
                context = x.new_zeros(batch, self.context_channels, height, width)
            elif context.shape[-2:] != (height, width):
                context = F.interpolate(
                    context, size=(height, width), mode="bilinear", align_corners=False
                )
            joined = torch.cat((x, hidden, context), dim=1)
        else:
            joined = torch.cat((x, hidden), dim=1)
        i, f, o, g = self.gates(joined).chunk(4, dim=1)
        cell = torch.sigmoid(f) * cell + torch.sigmoid(i) * torch.tanh(g)
        if self.cell_limit:
            # An unbounded cell integrates for as long as the camera runs; minutes of
            # live input drove it into saturation. Soft-bound it instead.
            cell = self.cell_limit * torch.tanh(cell / self.cell_limit)
        hidden = torch.sigmoid(o) * torch.tanh(cell)
        return hidden, cell


def positive_negative(value: Tensor) -> Tensor:
    return torch.cat((F.relu(value), F.relu(-value)), dim=1)


def warp_forward(image: Tensor, flow: Tensor) -> Tensor:
    """Sample a forward-displaced image; flow is in source-image pixels."""

    _, _, height, width = image.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype),
        torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(image.shape[0], -1, -1, -1)
    flow = flow.to(image.dtype)
    normalized = torch.stack(
        (flow[:, 0] * (2.0 / max(width - 1, 1)), flow[:, 1] * (2.0 / max(height - 1, 1))),
        dim=-1,
    )
    # Destination x + displacement samples source x: source = destination - flow.
    return F.grid_sample(
        image,
        grid - normalized,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def upsample_flow(flow: Tensor, size: tuple[int, int], scale: float) -> Tensor:
    return F.interpolate(flow, size=size, mode="bilinear", align_corners=False) * scale


class CorticalFlowNet(nn.Module):
    """A three-level predictive-coding hierarchy with persistent recurrent state.

    Level ``l`` sees the image at ``1 / strides[l]`` resolution. Its input is the
    signed error of its last prediction plus the current and changed input; its
    output is a displacement increment that refines the coarser level's motion.
    """

    def __init__(
        self,
        channels: tuple[int, int, int] = (16, 32, 64),
        strides: tuple[int, int, int] = (1, 4, 8),
        max_displacement: float = 24.0,
        residual_scale: float = 0.25,
        cell_limit: float = 4.0,
    ):
        super().__init__()
        if len(channels) != 3 or len(strides) != 3:
            raise ValueError("exactly three channel counts and strides are required")
        if strides[0] != 1 or any(b % a for a, b in zip(strides, strides[1:], strict=False)):
            raise ValueError("strides must start at 1 and each divide the next")
        self.channels = tuple(int(c) for c in channels)
        self.strides = tuple(int(s) for s in strides)
        self.max_displacement = float(max_displacement)
        self.residual_scale = float(residual_scale)
        self.cell_limit = float(cell_limit)
        # Coarsest level carries the full motion range; finer levels correct it by
        # up to two pixels of the level above.
        self.flow_bounds = (
            2.0 * self.strides[1],
            2.0 * self.strides[2] / self.strides[1],
            self.max_displacement / self.strides[2],
        )
        # Error populations (2x3), current input (3), and temporal change (3).
        inputs = 12
        self.cells = nn.ModuleList(
            [
                ConvLSTMCell(
                    inputs,
                    self.channels[level],
                    self.channels[level + 1] if level < 2 else 0,
                    self.cell_limit,
                )
                for level in range(3)
            ]
        )
        self.flow_heads = nn.ModuleList(
            [nn.Conv2d(self.channels[level], 2, 3, padding=1, bias=False) for level in range(3)]
        )
        self.residual_head = nn.Conv2d(self.channels[0], 3, 3, padding=1, bias=False)
        self.confidence_head = nn.Conv2d(self.channels[0], 1, 3, padding=1, bias=False)
        for head in (*self.flow_heads, self.residual_head, self.confidence_head):
            nn.init.zeros_(head.weight)

    def pyramid(self, frame: Tensor) -> list[Tensor]:
        return [frame if stride == 1 else F.avg_pool2d(frame, stride) for stride in self.strides]

    def forward(self, frame: Tensor, state: ModelState | None = None) -> StepOutput:
        if frame.ndim != 4 or frame.shape[1] != 3:
            raise ValueError("frame must have shape [batch, 3, height, width]")
        if frame.shape[-1] % self.strides[-1] or frame.shape[-2] % self.strides[-1]:
            raise ValueError(f"frame size must be divisible by {self.strides[-1]}")
        targets = self.pyramid(frame)
        if state is None:
            observed_error = None
            errors = [frame.new_zeros(t.shape[0], 6, *t.shape[-2:]) for t in targets]
            changes = [torch.zeros_like(t) for t in targets]
        else:
            observed_error = (frame - state.predicted[0]).abs().mean(dim=1, keepdim=True).detach()
            errors = [
                positive_negative(t - p) for t, p in zip(targets, state.predicted, strict=True)
            ]
            changes = [t - p for t, p in zip(targets, state.targets, strict=True)]

        recurrent: list[tuple[Tensor, Tensor]] = [(None, None)] * 3  # type: ignore[list-item]
        context: Tensor | None = None
        for level in (2, 1, 0):
            old_state = state.recurrent[level] if state is not None else None
            drive = torch.cat((errors[level], targets[level], changes[level]), dim=1)
            recurrent[level] = self.cells[level](drive, old_state, context)
            context = recurrent[level][0]

        flows: list[Tensor] = [None] * 3  # type: ignore[list-item]
        flow: Tensor | None = None
        for level in (2, 1, 0):
            increment = torch.tanh(self.flow_heads[level](recurrent[level][0]))
            increment = increment * self.flow_bounds[level]
            if flow is not None:
                ratio = self.strides[level + 1] / self.strides[level]
                flow = upsample_flow(flow, targets[level].shape[-2:], ratio)
                increment = increment + flow
            flow = increment
            flows[level] = flow

        if state is None:
            # Without history there is no evidence of motion: predict persistence.
            predicted = [t.clone() for t in targets]
            flows = [torch.zeros_like(f) for f in flows]
        else:
            predicted = [warp_forward(t, f) for t, f in zip(targets, flows, strict=True)]
            residual = torch.tanh(self.residual_head(recurrent[0][0])) * self.residual_scale
            predicted[0] = (predicted[0] + residual).clamp(0.0, 1.0)
        confidence = torch.sigmoid(self.confidence_head(recurrent[0][0]))

        next_state = ModelState(
            recurrent=recurrent,
            predicted=predicted,
            targets=targets,
            confidence=confidence,
            flow=flows[0],
        )
        return StepOutput(
            prediction=predicted[0],
            state=next_state,
            errors=errors,
            observed_error=observed_error,
            previous_confidence=state.confidence if state is not None else None,
            previous_flow=state.flow if state is not None else None,
        )

    def loss(
        self,
        output: StepOutput,
        level_weights: tuple[float, float] = (0.5, 0.5),
        confidence_weight: float = 0.1,
    ) -> Tensor:
        """Score a prediction only when its actual target has arrived."""

        if not output.has_ground_truth:
            return output.prediction.sum() * 0.0
        loss = output.errors[0].sum(dim=1).mean() / 3.0
        for level, weight in enumerate(level_weights, start=1):
            loss = loss + weight * output.errors[level].sum(dim=1).mean() / 3.0
        if output.previous_confidence is not None and output.observed_error is not None:
            expected_correctness = torch.exp(-10.0 * output.observed_error)
            loss = loss + confidence_weight * F.mse_loss(
                output.previous_confidence.float(), expected_correctness.float()
            )
        return loss

    def checkpoint_config(self) -> dict[str, object]:
        return {
            "channels": list(self.channels),
            "strides": list(self.strides),
            "max_displacement": self.max_displacement,
            "residual_scale": self.residual_scale,
            "cell_limit": self.cell_limit,
        }
