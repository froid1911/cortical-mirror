import subprocess
import sys

import cv2
import numpy as np

from cortical_mirror.cortex.engine import CortexEngine


def write_video(path, seed):
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 25, (160, 120))
    texture = rng.integers(0, 256, (120, 400, 3), dtype=np.uint8)
    for t in range(60):
        writer.write(np.ascontiguousarray(texture[:, 2 * t : 2 * t + 160]))
    writer.release()


def test_training_smoke_writes_loadable_checkpoint(tmp_path):
    data = tmp_path / "videos" / "Pan"
    data.mkdir(parents=True)
    write_video(data / "v_Pan_g10_c01.avi", 1)
    write_video(data / "v_Pan_g11_c01.avi", 2)
    output, log = tmp_path / "adult.pt", tmp_path / "train.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "tools/train_cortex.py",
            "--data",
            str(tmp_path / "videos"),
            "--streams",
            "2",
            "--frames",
            "200",
            "--log-every",
            "100",
            "--device",
            "cpu",
            "--output",
            str(output),
            "--log",
            str(log),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    engine = CortexEngine.load(output, learn=False)
    assert engine.age_frames == 200
    assert len(log.read_text().splitlines()) == 2


def test_training_an_adult_with_an_eye_saves_its_periphery(tmp_path):
    data = tmp_path / "videos" / "Pan"
    data.mkdir(parents=True)
    write_video(data / "v_Pan_g10_c01.avi", 1)
    output, log = tmp_path / "adult.pt", tmp_path / "train.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "tools/train_cortex.py",
            "--data",
            str(tmp_path / "videos"),
            "--phase",
            "3",
            "--streams",
            "2",
            "--frames",
            "30",
            "--log-every",
            "30",
            "--device",
            "cpu",
            "--output",
            str(output),
            "--log",
            str(log),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    engine = CortexEngine.load(output, learn=False)
    assert engine.phase == 3 and engine.periphery.age_frames == 30


def test_engine_exposes_the_code_it_observed_in_every_phase():
    import torch

    for phase, size in ((1, (96, 128)), (3, (192, 256))):
        engine = CortexEngine(phase=phase)
        engine.observe(torch.rand(2, 1, *size))
        assert engine.last_code.shape[-2:] == (96, 128)
