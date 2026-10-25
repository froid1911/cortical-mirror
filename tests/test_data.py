import cv2
import numpy as np
import pytest

from cortical_mirror.data import ClipConfig, SyntheticMotionDataset, VideoClipDataset, split_videos


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
    train_a, validation_a = split_videos(tmp_path, validation_percent=0)
    train_b, validation_b = split_videos(tmp_path, validation_percent=0)
    assert train_a == train_b
    assert validation_a == validation_b
    assert len(validation_a) == 1
    assert set(train_a).isdisjoint(validation_a)
    assert set(train_a + validation_a) == {tmp_path / name for name in ("a.avi", "b.avi", "c.avi")}


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
