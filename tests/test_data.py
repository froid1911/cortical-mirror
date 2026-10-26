import cv2
import numpy as np
import pytest
import torch

from cortical_mirror.data import (
    ClipConfig,
    SyntheticMotionDataset,
    VideoClipDataset,
    split_videos,
    teacher_flow,
)
from cortical_mirror.model import warp_forward


def test_synthetic_motion_is_repeatable_and_temporal():
    dataset = SyntheticMotionDataset(2, ClipConfig(width=64, height=48, frames=4))
    first = dataset[0]
    assert first.shape == (4, 3, 48, 64)
    assert first.dtype.is_floating_point
    assert 0.0 <= first.min() <= first.max() <= 1.0
    assert np.array_equal(first.numpy(), dataset[0].numpy())
    assert not np.array_equal(first[0].numpy(), first[-1].numpy())


def test_video_split_is_deterministic_and_has_no_leakage(tmp_path):
    for name in ("a.avi", "b.avi", "c.avi"):
        (tmp_path / name).write_bytes(b"")
    first = split_videos(tmp_path, validation_percent=0, test_percent=0)
    second = split_videos(tmp_path, validation_percent=0, test_percent=0)
    assert first == second
    assert len(first.validation) == 1
    assert set(first.train).isdisjoint(first.validation)
    everything = first.train + first.validation + first.test
    assert set(everything) == {tmp_path / name for name in ("a.avi", "b.avi", "c.avi")}


def test_ucf101_clips_from_one_source_video_never_cross_splits(tmp_path):
    folder = tmp_path / "Bowling"
    folder.mkdir()
    for group in range(1, 26):
        for clip in range(1, 5):
            (folder / f"v_Bowling_g{group:02d}_c{clip:02d}.avi").write_bytes(b"")
    split = split_videos(tmp_path)

    def groups(paths):
        return {path.stem.split("_")[2] for path in paths}

    assert groups(split.test) == {f"g{g:02d}" for g in range(1, 8)}
    assert groups(split.validation) == {"g08", "g09"}
    assert groups(split.train).isdisjoint(groups(split.validation) | groups(split.test))
    assert len(split.train) == 16 * 4


def test_teacher_flow_matches_renderer_convention():
    rng = np.random.default_rng(3)
    texture = cv2.GaussianBlur(rng.integers(0, 256, (64, 96, 3), dtype=np.uint8), (5, 5), 1.5)
    shifted = np.roll(texture, 3, axis=1)
    flow = teacher_flow(np.stack((texture, shifted)))
    assert flow.shape == (1, 2, 64, 96)
    centre = flow[0, :, 16:-16, 16:-16]
    assert abs(float(np.median(centre[0])) - 3.0) < 0.5
    assert abs(float(np.median(centre[1]))) < 0.5

    def tensor(image):
        return torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)

    warped = warp_forward(tensor(texture), torch.from_numpy(flow))
    moved_error = (warped - tensor(shifted))[..., 16:-16, 16:-16].abs().mean()
    copy_error = (tensor(texture) - tensor(shifted))[..., 16:-16, 16:-16].abs().mean()
    assert moved_error < 0.3 * copy_error


def test_unreadable_video_is_skipped_with_warning(tmp_path):
    bad = tmp_path / "bad.avi"
    bad.write_bytes(b"not a video")
    good = tmp_path / "good.avi"
    writer = cv2.VideoWriter(str(good), cv2.VideoWriter_fourcc(*"FFV1"), 10, (64, 48))
    for value in (20, 40, 60):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()
    dataset = VideoClipDataset(
        [bad, good],
        ClipConfig(width=64, height=48, frames=2, clips_per_video=1),
        deterministic=True,
    )
    with pytest.warns(RuntimeWarning, match="skipping unreadable video"):
        clip = dataset[0]
    assert clip.shape == (2, 3, 48, 64)
    assert 0.07 < float(clip[0].mean()) < 0.09
