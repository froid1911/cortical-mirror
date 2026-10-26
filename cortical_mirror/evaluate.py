"""Timestamp-aligned evaluation: score the prediction issued for this exact frame."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .engines import EnginePrediction, PredictiveEngine


def ssim(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(np.float32) / 255.0
    second = second.astype(np.float32) / 255.0
    c1, c2 = 0.01**2, 0.03**2
    values = []
    for channel in range(3):
        a, b = first[..., channel], second[..., channel]
        mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
        mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
        var_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
        var_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
        covariance = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
        values.append(
            float(
                np.mean(
                    ((2 * mu_a * mu_b + c1) * (2 * covariance + c2))
                    / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
                )
            )
        )
    return sum(values) / len(values)


class Perceptual:
    """LPIPS (Zhang et al., 2018): learned perceptual distance, lower is better.

    Blur that pixel MAE rewards scores badly here, so it guards against predictors
    that win MAE by smearing.
    """

    def __init__(self, device: str = "cpu"):
        import lpips
        import torch

        self.torch = torch
        self.device = device
        self.model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    def __call__(self, prediction: np.ndarray, actual: np.ndarray) -> float:
        def tensor(image: np.ndarray):
            rgb = image[..., ::-1].astype(np.float32) / 127.5 - 1.0
            return self.torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None].to(self.device)

        with self.torch.inference_mode():
            return float(self.model(tensor(prediction), tensor(actual)))


@dataclass
class Metrics:
    frames: int = 0
    absolute_error: float = 0.0
    squared_error: float = 0.0
    structural_similarity: float = 0.0
    perceptual_distance: float = 0.0
    perceptual: Perceptual | None = None

    def add(self, prediction: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
        difference = prediction.astype(np.float32) - actual.astype(np.float32)
        mae = float(np.mean(np.abs(difference)) / 255.0)
        mse = float(np.mean(difference**2) / (255.0**2))
        self.frames += 1
        self.absolute_error += mae
        self.squared_error += mse
        self.structural_similarity += ssim(prediction, actual)
        if self.perceptual is not None:
            self.perceptual_distance += self.perceptual(prediction, actual)
        return mae, mse

    def summary(self) -> dict[str, float | int]:
        if not self.frames:
            return {"frames": 0}
        mse = self.squared_error / self.frames
        summary = {
            "frames": self.frames,
            "mae": self.absolute_error / self.frames,
            "psnr_db": float(10.0 * np.log10(1.0 / max(mse, 1e-12))),
            "ssim": self.structural_similarity / self.frames,
        }
        if self.perceptual is not None:
            summary["lpips"] = self.perceptual_distance / self.frames
        return summary


def label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        result, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )
    return result


def comparison_view(
    actual: np.ndarray,
    due: EnginePrediction,
    copy: np.ndarray,
    future: EnginePrediction | None = None,
) -> np.ndarray:
    error = np.mean(np.abs(due.image.astype(np.float32) - actual.astype(np.float32)), axis=2)
    error = np.clip(error / 64.0, 0.0, 1.0)
    heatmap = cv2.applyColorMap(np.rint(error * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    top_right = future.image if future is not None else due.image
    bottom_left = due.image if future is not None else copy
    top = np.hstack(
        (
            label(actual, "camera now"),
            label(
                top_right, "predicted next" if future is not None else "prediction issued for now"
            ),
        )
    )
    bottom = np.hstack(
        (
            label(
                bottom_left, "prediction due now" if future is not None else "persistence baseline"
            ),
            label(heatmap, "absolute prediction error"),
        )
    )
    return np.vstack((top, bottom))


def evaluate_video(
    engine: PredictiveEngine,
    source: str | Path,
    output: str | Path | None = None,
    max_frames: int = 0,
    samples: int = 8,
    perceptual: Perceptual | None = None,
) -> dict:
    engine.reset()
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    output = Path(output) if output else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output / "comparison.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (engine.width * 2, engine.height * 2),
        )
    else:
        writer = None
    model_metrics = Metrics(perceptual=perceptual)
    copy_metrics = Metrics(perceptual=perceptual)
    wins = 0
    latencies = []
    pending: EnginePrediction | None = None
    previous_actual = None
    frame_index = 0
    start = time.perf_counter()
    while not max_frames or frame_index < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        actual = cv2.resize(frame, (engine.width, engine.height), interpolation=cv2.INTER_AREA)
        current = engine.predict(actual)
        latencies.append(current.inference_ms)
        if pending is not None:
            mae, _ = model_metrics.add(pending.image, actual)
            copy_mae, _ = copy_metrics.add(previous_actual, actual)
            wins += int(mae < copy_mae)
            view = comparison_view(actual, pending, previous_actual)
            if writer is not None:
                writer.write(view)
            if output and model_metrics.frames <= samples:
                index = model_metrics.frames
                cv2.imwrite(str(output / f"{index:03d}_actual.png"), actual)
                cv2.imwrite(str(output / f"{index:03d}_pred.png"), pending.image)
                cv2.imwrite(str(output / f"{index:03d}_copy.png"), previous_actual)
                cv2.imwrite(str(output / f"{index:03d}_comparison.png"), view)
        previous_actual = actual
        pending = current
        frame_index += 1
    capture.release()
    if writer is not None:
        writer.release()
    elapsed = time.perf_counter() - start
    steady_latencies = latencies[3:] if len(latencies) > 3 else latencies
    result = {
        "source": str(source),
        "width": engine.width,
        "height": engine.height,
        "frames_read": frame_index,
        "model": model_metrics.summary(),
        "copy": copy_metrics.summary(),
        "beat_copy": wins / model_metrics.frames if model_metrics.frames else 0.0,
        "cold_start_ms": latencies[0] if latencies else 0.0,
        "inference_ms_mean": float(np.mean(steady_latencies)) if steady_latencies else 0.0,
        "inference_ms_p95": float(np.percentile(steady_latencies, 95)) if steady_latencies else 0.0,
        "evaluation_wall_fps": frame_index / max(elapsed, 1e-6),
    }
    last_weights = getattr(engine, "last_weights", None)
    if last_weights is not None:
        result["expert_weights"] = {
            name: float(last_weights[..., index].mean()) for index, name in enumerate(engine.names)
        }
    if output:
        (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
