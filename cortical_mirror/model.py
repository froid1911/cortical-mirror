"""Hierarchical predictive coding with differentiable forward-motion rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class ModelState:
    """Everything carried from one observed frame to the next."""

    recurrent: list[tuple[Tensor, Tensor]]
    predicted: list[Tensor]
    confidence: Tensor

    def detached(self) -> "ModelState":
        return ModelState(
            recurrent=[(h.detach(), c.detach()) for h, c in self.recurrent],
            predicted=[p.detach() for p in self.predicted],
            confidence=self.confidence.detach(),
        )


@dataclass
class StepOutput:
    """Result of observing one frame and issuing a prediction for the next."""

    prediction: Tensor
    state: ModelState
    errors: list[Tensor]
    observed_error: Optional[Tensor]
    previous_confidence: Optional[Tensor]

    @property
    def has_ground_truth(self) -> bool:
        return self.observed_error is not None


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, context_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.context_channels = context_channels
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
        state: Optional[tuple[Tensor, Tensor]],
        context: Optional[Tensor],
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


class CorticalFlowNet(nn.Module):
    """A three-level predictive-coding hierarchy with persistent recurrent state.

    Each level predicts its next target representation. Observed deviations move
    upward as signed error populations; recurrent context moves downward. The
    finest level predicts forward flow, a residual, and expected correctness.
    """

    def __init__(
        self,
        channels: tuple[int, int, int] = (24, 48, 96),
        max_displacement: float = 24.0,
        residual_scale: float = 0.25,
    ):
        super().__init__()
        if len(channels) != 3:
            raise ValueError("exactly three channel counts are required")
        self.channels = tuple(channels)
        self.max_displacement = float(max_displacement)
        self.residual_scale = float(residual_scale)
        self.target_channels = (3, channels[1], channels[2])

        self.target_convs = nn.ModuleList(
            [
                nn.Conv2d(2 * self.target_channels[level], self.target_channels[level + 1], 3, 2, 1, bias=False)
                for level in range(2)
            ]
        )
        self.cells = nn.ModuleList(
            [
                ConvLSTMCell(
                    2 * self.target_channels[level],
                    channels[level],
                    channels[level + 1] if level < 2 else 0,
                )
                for level in range(3)
            ]
        )
        self.representation_predictors = nn.ModuleList(
            [nn.Conv2d(channels[level], self.target_channels[level], 1, bias=False) for level in range(1, 3)]
        )
        self.flow_head = nn.Conv2d(channels[0], 2, 3, padding=1, bias=False)
        self.residual_head = nn.Conv2d(channels[0], 3, 3, padding=1, bias=False)
        self.confidence_head = nn.Conv2d(channels[0], 1, 3, padding=1, bias=False)
        for head in (self.flow_head, self.residual_head, self.confidence_head):
            nn.init.zeros_(head.weight)

    def forward(self, frame: Tensor, state: Optional[ModelState] = None) -> StepOutput:
        if frame.ndim != 4 or frame.shape[1] != 3:
            raise ValueError("frame must have shape [batch, 3, height, width]")
        previous = state.predicted if state is not None else None
        previous_confidence = state.confidence if state is not None else None
        observed_error = (
            (frame - previous[0]).abs().mean(dim=1, keepdim=True).detach()
            if previous is not None
            else None
        )

        targets: list[Tensor] = []
        errors: list[Tensor] = []
        target = frame if previous is not None else torch.zeros_like(frame)
        for level in range(3):
            if level:
                target = self.target_convs[level - 1](errors[-1])
            targets.append(target)
            predicted = previous[level] if previous is not None else torch.zeros_like(target)
            errors.append(positive_negative(target - predicted))

        recurrent: list[tuple[Tensor, Tensor]] = [(None, None)] * 3  # type: ignore[list-item]
        context: Optional[Tensor] = None
        for level in (2, 1, 0):
            old_state = state.recurrent[level] if state is not None else None
            hidden, cell = self.cells[level](errors[level], old_state, context)
            recurrent[level] = (hidden, cell)
            context = hidden

        flow = torch.tanh(self.flow_head(recurrent[0][0])) * self.max_displacement
        residual = torch.tanh(self.residual_head(recurrent[0][0])) * self.residual_scale
        prediction = (warp_forward(frame, flow) + residual).clamp(0.0, 1.0)
        confidence = torch.sigmoid(self.confidence_head(recurrent[0][0]))

        predicted = [prediction]
        predicted.extend(
            self.representation_predictors[level - 1](recurrent[level][0]) for level in (1, 2)
        )
        next_state = ModelState(recurrent=recurrent, predicted=predicted, confidence=confidence)
        return StepOutput(
            prediction=prediction,
            state=next_state,
            errors=errors,
            observed_error=observed_error,
            previous_confidence=previous_confidence,
        )

    def loss(
        self,
        output: StepOutput,
        level_weights: tuple[float, float] = (0.1, 0.05),
        confidence_weight: float = 0.1,
    ) -> Tensor:
        """Score a prediction only when its actual target has arrived."""

        if not output.has_ground_truth:
            return output.prediction.sum() * 0.0
        pixel_mae = output.errors[0].sum(dim=1).mean() / 3.0
        loss = pixel_mae
        for level, weight in enumerate(level_weights, start=1):
            feature_mae = output.errors[level].sum(dim=1).mean() / self.target_channels[level]
            loss = loss + weight * feature_mae
        if output.previous_confidence is not None and output.observed_error is not None:
            expected_correctness = torch.exp(-10.0 * output.observed_error)
            loss = loss + confidence_weight * F.mse_loss(
                output.previous_confidence, expected_correctness
            )
        return loss

    def checkpoint_config(self) -> dict[str, object]:
        return {
            "channels": list(self.channels),
            "max_displacement": self.max_displacement,
            "residual_scale": self.residual_scale,
        }
