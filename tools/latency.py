"""Measure mirror inference latency and live-learning throughput on this machine.

Frames are fed at the clip's own rate, as a camera would deliver them, through
the same engines `cortical-mirror demo` builds. Each configuration reports mean
and 95th-percentile inference time after a warm-up, and how many learner steps
per second (and per frame) the background thread achieved meanwhile.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from cortical_mirror.dmvfn import DMVFNEngine
from cortical_mirror.engines import CopyEngine, CorticalEngine, resolve_device
from cortical_mirror.ensemble import AdaptiveEngine


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def load_frames(path: Path, seconds: float) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (256, 192), interpolation=cv2.INTER_AREA))
    capture.release()
    repeats = int(np.ceil(seconds * fps / len(frames)))
    return (frames * repeats)[: int(seconds * fps)], fps


def build(args, device: str, scratch: bool, learn: bool):
    mixed = False  # the adaptive mirror runs its experts in FP32
    cortical = (
        CorticalEngine.fresh(device=device, mixed_precision=mixed)
        if scratch
        else CorticalEngine.from_checkpoint(args.checkpoint, device=device, mixed_precision=mixed)
    )
    experts = {"copy": CopyEngine(), "cortical": cortical}
    if device != "cpu" and args.dmvfn_checkpoint:
        experts["dmvfn"] = DMVFNEngine(args.dmvfn_checkpoint, device=device, mixed_precision=False)
    learner = None
    if learn:
        learner = cortical.enable_learning(
            learning_rate=1e-3 if scratch else 2e-4,
            steps_per_frame=10.0,
            anchor_weight=0.0 if scratch else 1e-2,
        )
    return AdaptiveEngine(experts, temperature=0.002, adaptation=0.1), learner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dmvfn-checkpoint", type=Path)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = resolve_device(args.device)
    frames, fps = load_frames(args.video, args.seconds)

    results = []
    for label, scratch, learn in (
        ("pretrained, no learning", False, False),
        ("pretrained, learning", False, True),
        ("from scratch, learning", True, True),
    ):
        engine, learner = build(args, device, scratch, learn)
        times = []
        started = time.perf_counter()
        for frame in frames:
            tick = time.perf_counter()
            times.append(engine.predict(frame).inference_ms)
            time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - tick)))
        wall = time.perf_counter() - started
        steps = learner.stats().steps if learner else 0
        if learner:
            learner.stop()
        steady = np.array(times[int(fps) :])
        result = {
            "configuration": label,
            "experts": list(engine.names),
            "inference_ms_mean": round(float(steady.mean()), 1),
            "inference_ms_p95": round(float(np.percentile(steady, 95)), 1),
            "learner_steps_per_second": round(steps / wall, 1),
            "learner_steps_per_frame": round(steps / len(frames), 2),
        }
        results.append(result)
        print(json.dumps(result), flush=True)

    hardware = torch.cuda.get_device_name(0) if device != "cpu" else cpu_name()
    report = {
        "device": device,
        "hardware": hardware,
        "cpu_threads": torch.get_num_threads(),
        "video": args.video.name,
        "fps": fps,
        "seconds": args.seconds,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
