"""Raise MT's motion cells on hours of video, over a frozen V1 adult.

Several streams watch different training-split videos at once, frame by frame,
moving to a new random video when one ends (as for the cortex adults). The rule is
exactly the newborn's. Every --log-every steps the learned displacements are
logged; the checkpoint is saved then too.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from train_cortex import Streams

from cortical_mirror.cortex.engine import RETINA_SIZE
from cortical_mirror.cortex.mt import MotionCortex
from cortical_mirror.data import split_videos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--v1", type=Path, required=True, help="the phase 1 cortex adult")
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--types", type=int, default=16)
    parser.add_argument(
        "--fast-share",
        type=float,
        default=0.0,
        help="share of streams that skip every other frame (doubling their speeds)",
    )
    parser.add_argument("--log-every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()

    split = split_videos(args.data)
    cortex = MotionCortex(v1=args.v1, types=args.types, seed=args.seed, device=args.device)
    fast = round(args.fast_share * args.streams)
    strides = [2] * fast + [1] * (args.streams - fast)
    streams = Streams(split.train, args.streams, args.seed, strides=strides)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    started, step = time.time(), 0
    with args.log.open("w", encoding="utf-8") as log:
        while time.time() < started + args.hours * 3600:
            cortex.step(streams.frames(cortex.device, RETINA_SIZE), learn=True)
            step += 1
            if step % args.log_every == 0:
                fine = cortex.fine.displacements()
                record = {
                    "step": step,
                    "stream_frames": step * args.streams,
                    "seconds": round(time.time() - started),
                    "fine_lengths": [round(float(v), 2) for v in fine.norm(dim=1)],
                    "fine_angles": [
                        round(float(torch.atan2(d[1], d[0]).rad2deg()), 0) for d in fine
                    ],
                    "coarse_lengths": [
                        round(float(v), 2) for v in cortex.coarse.displacements().norm(dim=1)
                    ],
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                cortex.save(args.output)
    cortex.save(args.output)


if __name__ == "__main__":
    main()
