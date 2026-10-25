"""Command-line interface for live demos and aligned offline evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .classical import OpticalFlowEngine
from .dmvfn import DMVFNEngine
from .engines import CopyEngine, CorticalEngine
from .ensemble import AdaptiveEngine
from .evaluate import evaluate_video
from .server import serve


def add_engine_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--engine", choices=("cortical", "dmvfn", "adaptive", "flow", "copy"), required=True
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="cortical checkpoint, or standalone DMVFN checkpoint"
    )
    parser.add_argument(
        "--dmvfn-checkpoint", type=Path, help="DMVFN checkpoint used by --engine adaptive"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--no-amp", action="store_true")


def create_engine(args):
    if args.engine in ("cortical", "dmvfn", "adaptive") and not args.checkpoint:
        raise SystemExit(f"--engine {args.engine} requires --checkpoint")
    if args.engine == "adaptive":
        if not args.dmvfn_checkpoint:
            raise SystemExit("--engine adaptive requires --dmvfn-checkpoint")
        cortical = CorticalEngine.from_checkpoint(
            args.checkpoint, device=args.device, mixed_precision=not args.no_amp
        )
        dmvfn = DMVFNEngine(
            args.dmvfn_checkpoint,
            device=args.device,
            width=cortical.width,
            height=cortical.height,
            mixed_precision=not args.no_amp,
        )
        return AdaptiveEngine(
            {
                "copy": CopyEngine(cortical.width, cortical.height),
                "cortical": cortical,
                "dmvfn": dmvfn,
            }
        )
    if args.engine == "cortical":
        return CorticalEngine.from_checkpoint(
            args.checkpoint, device=args.device, mixed_precision=not args.no_amp
        )
    if args.engine == "dmvfn":
        return DMVFNEngine(
            args.checkpoint,
            device=args.device,
            width=args.width,
            height=args.height,
            mixed_precision=not args.no_amp,
        )
    if args.engine == "flow":
        return OpticalFlowEngine(width=args.width, height=args.height)
    return CopyEngine(width=args.width, height=args.height)


def main() -> None:
    parser = argparse.ArgumentParser(prog="cortical-mirror", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    live = commands.add_parser("serve", help="serve the live mirror in a browser")
    add_engine_arguments(live)
    live.add_argument("--source", default="0", help="camera index or video path")
    live.add_argument("--host", default="127.0.0.1")
    live.add_argument("--port", type=int, default=8080)
    live.add_argument("--no-loop", action="store_true", help="stop at the end of a video source")
    live.add_argument("--max-frames", type=int, default=0)

    offline = commands.add_parser(
        "evaluate", help="evaluate predictions against their exact target frames"
    )
    add_engine_arguments(offline)
    offline.add_argument("--source", required=True, help="video path")
    offline.add_argument("--output", type=Path, required=True)
    offline.add_argument("--max-frames", type=int, default=120)
    offline.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()

    engine = create_engine(args)
    if args.command == "serve":
        serve(
            engine,
            source=args.source,
            engine_name=args.engine,
            host=args.host,
            port=args.port,
            loop=not args.no_loop,
            max_frames=args.max_frames,
        )
    else:
        result = evaluate_video(
            engine,
            args.source,
            output=args.output,
            max_frames=args.max_frames,
            samples=args.samples,
        )
        print(json.dumps({args.engine: result}, indent=2))


if __name__ == "__main__":
    main()
