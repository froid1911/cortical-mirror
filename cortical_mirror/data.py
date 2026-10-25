"""Video clips for self-supervised next-frame training."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv"}


@dataclass(frozen=True)
class ClipConfig:
    width: int = 256
    height: int = 192
    frames: int = 6
    stride: int = 1
    clips_per_video: int = 4


def split_videos(root: str | Path, validation_percent: int = 5) -> tuple[list[Path], list[Path]]:
    root = Path(root)
    videos = sorted(path for path in root.rglob("*") if path.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        raise FileNotFoundError(f"no video files found under {root}")
    train, validation = [], []
    for path in videos:
        relative = str(path.relative_to(root)).encode()
        bucket = int.from_bytes(hashlib.sha256(relative).digest()[:2], "big") % 100
        (validation if bucket < validation_percent else train).append(path)
    if not validation:
        validation.append(train.pop())
    return train, validation


def read_clip(path: Path, config: ClipConfig, start: int) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for index in range(config.frames):
        if index:
            for _ in range(config.stride - 1):
                capture.grab()
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"cannot decode frame {start + index * config.stride} in {path}")
        frame = cv2.resize(frame, (config.width, config.height), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    capture.release()
    array = np.stack(frames).astype(np.float32) / 255.0
    return torch.from_numpy(array.transpose(0, 3, 1, 2).copy())


class VideoClipDataset(Dataset):
    def __init__(self, videos: list[Path], config: ClipConfig, deterministic: bool = False):
        self.videos = videos
        self.config = config
        self.deterministic = deterministic
        if not videos:
            raise ValueError("dataset needs at least one video")

    def __len__(self) -> int:
        return len(self.videos) * self.config.clips_per_video

    def __getitem__(self, index: int) -> torch.Tensor:
        video_index = index % len(self.videos)
        path = self.videos[video_index]
        capture = cv2.VideoCapture(str(path))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        span = (self.config.frames - 1) * self.config.stride + 1
        max_start = max(0, count - span)
        if self.deterministic:
            start = 0 if max_start == 0 else (index * 37) % (max_start + 1)
        else:
            start = random.randint(0, max_start)
        return read_clip(path, self.config, start)


class SyntheticMotionDataset(Dataset):
    """Deterministic textured motion for tests and pipeline smoke runs only."""

    def __init__(self, length: int, config: ClipConfig):
        self.length = length
        self.config = config

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        cfg = self.config
        rng = np.random.default_rng(index)
        height, width = cfg.height, cfg.width
        background = rng.normal(0.35, 0.04, (height, width, 1)).astype(np.float32)
        background = np.repeat(background, 3, axis=2)
        background += np.linspace(0.0, 0.25, width, dtype=np.float32)[None, :, None]
        size = int(rng.integers(max(12, height // 8), max(20, height // 4)))
        velocity = rng.integers(-4, 5, size=2)
        velocity[velocity == 0] = 2
        position = rng.integers(0, [max(width - size - 12, 1), max(height - size - 12, 1)]).astype(
            float
        )
        position += np.array([6.0, 6.0])
        texture = rng.random((size, size, 3), dtype=np.float32) * 0.8 + 0.2
        frames = []
        x, y = position
        vx, vy = velocity.astype(float)
        for _ in range(cfg.frames):
            frame = background.copy()
            x0, y0 = int(round(x)), int(round(y))
            x1, y1 = x0 + size, y0 + size
            sx0, sy0 = max(0, -x0), max(0, -y0)
            dx0, dy0 = max(0, x0), max(0, y0)
            dx1, dy1 = min(width, x1), min(height, y1)
            if dx0 < dx1 and dy0 < dy1:
                frame[dy0:dy1, dx0:dx1] = texture[sy0 : sy0 + (dy1 - dy0), sx0 : sx0 + (dx1 - dx0)]
            cv2.rectangle(frame, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), (0.9, 0.9, 0.9), 1)
            frame += rng.normal(0.0, 0.01, frame.shape).astype(np.float32)
            frames.append(np.clip(frame, 0.0, 1.0))
            x += vx
            y += vy
            if x < 0 or x + size >= width:
                vx *= -1
                x = float(np.clip(x, 0, width - size))
            if y < 0 or y + size >= height:
                vy *= -1
                y = float(np.clip(y, 0, height - size))
        array = np.stack(frames).astype(np.float32)
        return torch.from_numpy(array.transpose(0, 3, 1, 2).copy())
