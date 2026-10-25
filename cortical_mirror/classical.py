"""Classical coarse-to-fine optical-flow prediction with online gain adaptation."""

from __future__ import annotations

import time

import cv2
import numpy as np

from .engines import EnginePrediction, _resize

EPS = 1e-6


def dense_flow(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(previous_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.1, 0)


def splat(image: np.ndarray, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    dx, dy = xx + vectors[..., 0], yy + vectors[..., 1]
    x0, y0 = np.floor(dx).astype(np.int32), np.floor(dy).astype(np.int32)
    fx, fy = dx - x0, dy - y0
    accumulation = np.zeros((height, width, 3), np.float32)
    weight = np.zeros((height, width), np.float32)
    for oy in (0, 1):
        for ox in (0, 1):
            tx, ty = x0 + ox, y0 + oy
            valid = (tx >= 0) & (tx < width) & (ty >= 0) & (ty < height)
            if not valid.any():
                continue
            contribution = ((1 - fx) if ox == 0 else fx) * ((1 - fy) if oy == 0 else fy)
            indices = ty[valid].astype(np.int64) * width + tx[valid]
            values = contribution[valid]
            for channel in range(3):
                accumulation[..., channel] += np.bincount(
                    indices,
                    weights=image[yy[valid], xx[valid], channel] * values,
                    minlength=height * width,
                ).reshape(height, width)
            weight += np.bincount(indices, weights=values, minlength=height * width).reshape(
                height, width
            )
    return accumulation, np.minimum(weight, 1.0)


def extrapolate(
    current: np.ndarray,
    previous: np.ndarray,
    gain: float,
    acceleration: float,
    height: int,
    width: int,
):
    vectors = gain * current + acceleration * (current - previous)
    maximum = 0.25 * min(height, width)
    magnitude = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.where(magnitude > maximum, vectors * maximum / np.maximum(magnitude, EPS), vectors)


def fill_coarse(image: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    accumulation, weight = splat(image, vectors)
    output = accumulation / np.maximum(weight, EPS)[..., None]
    holes = (weight < 0.25).astype(np.uint8) * 255
    if holes.any():
        output = cv2.inpaint(
            np.clip(output, 0, 255).astype(np.uint8), holes, 4, cv2.INPAINT_TELEA
        ).astype(np.float32)
    return output


def predict_frame(
    current: np.ndarray, flow: np.ndarray, previous: np.ndarray, gain: float, acceleration: float
):
    height, width = current.shape[:2]
    vectors = extrapolate(flow, previous, gain, acceleration, height, width)
    accumulation, weight = splat(current.astype(np.float32), vectors)
    fine = accumulation / np.maximum(weight, EPS)[..., None]
    small = cv2.resize(current, (width // 4, height // 4), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )
    coarse_vectors = (
        cv2.resize(vectors, (width // 4, height // 4), interpolation=cv2.INTER_AREA) / 4.0
    )
    coarse = fill_coarse(small, coarse_vectors)
    coarse = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_LINEAR)
    blend = cv2.GaussianBlur(np.clip(weight, 0, 1), (9, 9), 0)[..., None]
    return np.clip(fine * blend + coarse * (1 - blend), 0, 255)


def best_gain(flow: np.ndarray, previous: np.ndarray, source: np.ndarray, actual: np.ndarray):
    best = (float("inf"), 1.0, 0.0)
    height, width = source.shape[:2]
    for gain in (0.5, 0.75, 1.0, 1.25, 1.5):
        for acceleration in (-0.5, 0.0, 0.5):
            vectors = extrapolate(flow, previous, gain, acceleration, height, width)
            accumulation, weight = splat(source, vectors)
            prediction = accumulation / np.maximum(weight, EPS)[..., None]
            mask = weight > 0.1
            if mask.any():
                error = float(np.mean(np.abs(prediction - actual)[mask]))
                if error < best[0]:
                    best = error, gain, acceleration
    return best[1], best[2]


def coarse(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


class OpticalFlowEngine:
    def __init__(self, width: int = 256, height: int = 192):
        self.width, self.height = width, height
        self.reset()

    def reset(self) -> None:
        self.frames = []
        self.flow = None
        self.previous_flow = None
        self.pending = None
        self.gain, self.acceleration = 1.0, 0.0

    def predict(self, frame_bgr: np.ndarray) -> EnginePrediction:
        start = time.perf_counter()
        current = _resize(frame_bgr, self.width, self.height)
        observed = (
            np.mean(np.abs(current.astype(np.float32) - self.pending), axis=2)
            if self.pending is not None
            else None
        )
        if len(self.frames) >= 2 and self.flow is not None and self.previous_flow is not None:
            small_width, small_height = self.width // 4, self.height // 4
            gain, acceleration = best_gain(
                coarse(self.flow, small_width, small_height) / 4.0,
                coarse(self.previous_flow, small_width, small_height) / 4.0,
                coarse(self.frames[-1], small_width, small_height).astype(np.float32),
                coarse(current, small_width, small_height).astype(np.float32),
            )
            self.gain = 0.7 * self.gain + 0.3 * gain
            self.acceleration = 0.7 * self.acceleration + 0.3 * acceleration

        self.frames.append(current)
        self.frames = self.frames[-2:]
        if len(self.frames) >= 2:
            self.previous_flow, self.flow = self.flow, dense_flow(self.frames[-2], self.frames[-1])
        if self.flow is None:
            prediction = current.astype(np.float32)
        else:
            previous = (
                self.previous_flow if self.previous_flow is not None else np.zeros_like(self.flow)
            )
            prediction = predict_frame(current, self.flow, previous, self.gain, self.acceleration)
        self.pending = prediction
        return EnginePrediction(
            image=np.rint(prediction).astype(np.uint8),
            confidence=np.full((self.height, self.width), np.nan, dtype=np.float32),
            observed_error=observed,
            inference_ms=(time.perf_counter() - start) * 1000.0,
        )
