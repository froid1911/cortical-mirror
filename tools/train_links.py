"""Learn the association field (protocol 19's rule) from MT integrated over a gap."""

import argparse
import time
from collections import deque
from pathlib import Path

import torch
from slots_report import view  # tools/ is on the path when run as a script

from cortical_mirror.cortex.grouping import AssociationField
from cortical_mirror.cortex.mt import MotionCortex
from cortical_mirror.scene.world import SpriteWorld


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mt", type=Path, required=True, help="the v0.9 MT adult")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs=2, default=(900, 950))
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--gap", type=int, default=4)
    parser.add_argument("--smoothing", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fast", action="store_true", help="MT reads V1's feedforward sweep")
    args = parser.parse_args()

    cortex = MotionCortex.load(args.mt, device=args.device, fast=args.fast)
    cortex.fine.smoothing = cortex.coarse.smoothing = args.smoothing
    field = AssociationField(types=cortex.hierarchy.configs[0].cells, device=args.device)
    for seed in range(*args.seeds):
        world, started = SpriteWorld(seed=seed), time.time()
        past = deque(maxlen=args.gap + 1)
        for _ in range(args.frames):
            world.step()
            past.append(cortex.v1(view(world, cortex.device)))
            if len(past) > args.gap:
                velocity = cortex.between(past[0], past[-1])[0] / args.gap
                field.learn(past[-1][0], velocity)
        print(f"seed {seed} in {time.time() - started:.1f}s", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"links": field.links.cpu(), "offsets": field.offsets}, args.output)


if __name__ == "__main__":
    main()
