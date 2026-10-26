"""Video clips for self-supervised next-frame training."""

from __future__ import annotations

import hashlib
import random
import re
import warnings
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


UCF101_NAME = re.compile(r"^v_(?P<action>[A-Za-z]+)_g(?P<group>\d+)_c\d+$")
# Official UCF101 split 1 tests on groups 1-7. Clips in one group are cut from the
# same source video, so groups, never clips, are assigned to a split.
UCF101_TEST_GROUPS = frozenset(range(1, 8))
UCF101_VALIDATION_GROUPS = frozenset({8, 9})


@dataclass(frozen=True)
class VideoSplit:
    train: list[Path]
    validation: list[Path]
    test: list[Path]


def source_group(path: Path) -> str:
    """Identify the source recording a clip was cut from."""

    match = UCF101_NAME.match(path.stem)
    if match:
        return f"{match['action']}/g{int(match['group']):02d}"
    return str(path.with_suffix(""))


def split_videos(
    root: str | Path, validation_percent: int = 5, test_percent: int = 5
) -> VideoSplit:
    """Assign whole source recordings to train, validation, or test.

    UCF101 uses the official split-1 test groups and two held-back training groups
    for validation. Other folders are split by hashing each file's source group.
    Checkpoints and routing settings may be chosen on validation, never on test.
    """

    root = Path(root)
    videos = sorted(path for path in root.rglob("*") if path.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        raise FileNotFoundError(f"no video files found under {root}")
    split = VideoSplit([], [], [])
    for path in videos:
        match = UCF101_NAME.match(path.stem)
        if match:
            group = int(match["group"])
            if group in UCF101_TEST_GROUPS:
                split.test.append(path)
            elif group in UCF101_VALIDATION_GROUPS:
                split.validation.append(path)
            else:
                split.train.append(path)
            continue
        relative = source_group(path.relative_to(root)).encode()
        bucket = int.from_bytes(hashlib.sha256(relative).digest()[:2], "big") % 100
        if bucket < test_percent:
            split.test.append(path)
        elif bucket < test_percent + validation_percent:
            split.validation.append(path)
        else:
            split.train.append(path)
    if not split.validation and len(split.train) > 1:
        split.validation.append(split.train.pop())
    return split


def teacher_flow(frames: np.ndarray) -> np.ndarray:
    """Classical flow behind each transition, in the renderer's warp convention.

    ``frames`` is ``[T, H, W, 3]`` uint8 RGB. Returns ``[T-1, 2, H, W]`` such that
    ``warp_forward(frame[t], flow[t])`` approximates ``frame[t+1]``.
    """

    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    flows = []
    for current, following in zip(gray, gray[1:], strict=False):
        # following(x) ~= current(x + backward(x)); the renderer samples x - flow.
        backward = cv2.calcOpticalFlowFarneback(following, current, None, 0.5, 3, 15, 3, 5, 1.1, 0)
        flows.append(-backward.transpose(2, 0, 1))
    return np.stack(flows).astype(np.float32)


def read_clip(path: Path, config: ClipConfig, start: int) -> np.ndarray:
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
    return np.stack(frames)


def to_tensor(frames: np.ndarray) -> torch.Tensor:
    array = frames.astype(np.float32) / 255.0
    return torch.from_numpy(array.transpose(0, 3, 1, 2).copy())


class VideoClipDataset(Dataset):
    def __init__(
        self,
        videos: list[Path],
        config: ClipConfig,
        deterministic: bool = False,
        with_flow: bool = False,
    ):
        self.videos = videos
        self.config = config
        self.deterministic = deterministic
        self.with_flow = with_flow
        self.unreadable: set[Path] = set()
        if not videos:
            raise ValueError("dataset needs at least one video")

    def __len__(self) -> int:
        return len(self.videos) * self.config.clips_per_video

    def __getitem__(self, index: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        frames = self._frames(index)
        if not self.deterministic and random.random() < 0.5:
            # A mirrored clip is equally plausible physics, and the demo is a mirror.
            frames = np.ascontiguousarray(frames[:, :, ::-1])
        if self.with_flow:
            return to_tensor(frames), torch.from_numpy(teacher_flow(frames))
        return to_tensor(frames)

    def _frames(self, index: int) -> np.ndarray:
        last_error = None
        for offset in range(min(3, len(self.videos))):
            video_index = (index + offset) % len(self.videos)
            path = self.videos[video_index]
            if path in self.unreadable:
                continue
            capture = cv2.VideoCapture(str(path))
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            span = (self.config.frames - 1) * self.config.stride + 1
            max_start = max(0, count - span)
            if self.deterministic:
                start = 0 if max_start == 0 else (index * 37) % (max_start + 1)
            else:
                start = random.randint(0, max_start)
            try:
                return read_clip(path, self.config, start)
            except RuntimeError as error:
                last_error = error
                if start:
                    try:
                        return read_clip(path, self.config, 0)
                    except RuntimeError as fallback_error:
                        last_error = fallback_error
                warnings.warn(
                    f"skipping unreadable video {path}: {last_error}", RuntimeWarning, stacklevel=2
                )
                self.unreadable.add(path)
        raise RuntimeError("could not decode a clip after three video attempts") from last_error


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
