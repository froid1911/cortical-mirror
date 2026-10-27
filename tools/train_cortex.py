"""Raise an adult cortex: the newborn's network and rules, on hours of video.

Several streams watch different training-split videos at once, each one frame at
a time in order, moving to a new random video when one ends. One step is one
frame for every stream; the update is averaged over streams, as if one cortex
had lived through all of them. The rules are exactly the newborn's.
"""

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import torch

from cortical_mirror.cortex.engine import CortexEngine
from cortical_mirror.cortex.phenomena import orientation_tuning
from cortical_mirror.cortex.retina import frame_to_gray
from cortical_mirror.data import split_videos


class Streams:
    def __init__(self, videos: list[Path], count: int, seed: int):
        self.videos = videos
        self.rng = random.Random(seed)
        self.captures = [self._open() for _ in range(count)]

    def _open(self):
        return cv2.VideoCapture(str(self.rng.choice(self.videos)))

    def frames(self, device, size: tuple[int, int]) -> torch.Tensor:
        out = []
        for index, capture in enumerate(self.captures):
            ok, frame = capture.read()
            while not ok:
                capture.release()
                capture = self.captures[index] = self._open()
                ok, frame = capture.read()
            out.append(frame_to_gray(frame, *size, device))
        return torch.cat(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--frames", type=int, default=0, help="steps to run (0 = until --hours)")
    parser.add_argument("--hours", type=float, default=3.0)
    parser.add_argument("--log-every", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()

    split = split_videos(args.data)
    videos = split.train or split.validation
    engine = CortexEngine(device=args.device, learn=True, phase=args.phase, seed=args.seed)
    engine.streams = args.streams
    streams = Streams(videos, args.streams, args.seed)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    deadline = started + args.hours * 3600
    step = 0
    with args.log.open("w", encoding="utf-8") as log:
        while (args.frames and step < args.frames) or (not args.frames and time.time() < deadline):
            world = streams.frames(engine.device, engine.input_size)
            result = engine.observe(world)
            code = engine.last_code
            step += 1
            if step % args.log_every == 0 or step == args.frames:
                activity = result.activities[0]
                tuning = orientation_tuning(engine.hierarchy)
                record = {
                    "step": step,
                    "seconds": round(time.time() - started, 1),
                    "stream_frames": step * args.streams,
                    "reconstruction_error": float(
                        result.errors[0].square().mean() / code.square().mean().clamp_min(1e-12)
                    ),
                    "prediction_over_persistence": result.code_error
                    / max(result.persistence_error, 1e-12),
                    "prediction_over_stillness": result.code_error
                    / max(result.stillness_error, 1e-12),
                    "mean_rate": float((activity > 0).float().mean()),
                    "median_osi": tuning["median_osi"],
                    "fraction_selective": tuning["fraction_selective"],
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                engine.save(args.output)
    engine.save(args.output)


if __name__ == "__main__":
    main()
