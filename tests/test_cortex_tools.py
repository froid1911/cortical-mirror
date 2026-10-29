import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from cortical_mirror.cortex.engine import CortexEngine


def phenomena_tool():
    spec = importlib.util.spec_from_file_location("phenomena_tool", "tools/phenomena.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_raising_on_an_unreadable_video_fails_instead_of_hanging(tmp_path):
    with pytest.raises(SystemExit, match="cannot read"):
        phenomena_tool().raise_on(CortexEngine(), tmp_path / "missing.avi", 10)


def test_phase_that_contradicts_the_checkpoint_is_refused(tmp_path):
    CortexEngine(phase=1).save(tmp_path / "phase1.pt")
    result = subprocess.run(
        [
            sys.executable,
            "tools/phenomena.py",
            "--checkpoint",
            str(tmp_path / "phase1.pt"),
            "--phase",
            "2",
            "--device",
            "cpu",
            "--output",
            str(tmp_path / "out.json"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode != 0
    assert "phase" in result.stderr
    assert not Path(tmp_path / "out.json").exists()


def train_cortex_tool():
    spec = importlib.util.spec_from_file_location("train_cortex_tool", "tools/train_cortex.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_stream_with_stride_two_sees_every_other_frame(tmp_path):
    import cv2
    import numpy as np

    path = tmp_path / "count.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (32, 24), False)
    for value in range(0, 200, 20):
        writer.write(np.full((24, 32), value, np.uint8))
    writer.release()
    streams = train_cortex_tool().Streams([path], 2, seed=0, strides=[1, 2])
    seen = [streams.frames("cpu", (32, 24))[:, 0].mean((1, 2)) * 255 for _ in range(3)]
    plain = [round(float(s[0]) / 20) for s in seen]
    skipping = [round(float(s[1]) / 20) for s in seen]
    assert plain == [0, 1, 2] and skipping == [1, 3, 5]
