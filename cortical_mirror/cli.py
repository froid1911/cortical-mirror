"""Command-line interface for live demos and aligned offline evaluation."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import cv2

from . import __version__
from .classical import OpticalFlowEngine
from .dmvfn import DMVFNEngine
from .engines import CopyEngine, CorticalEngine, resolve_device
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
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:N, or cpu")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--no-amp", action="store_true")
    # Chosen on the validation split (reports/tune/); the test split was never used.
    parser.add_argument("--temperature", type=float, default=0.002)
    parser.add_argument("--adaptation", type=float, default=0.1)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument(
        "--experts",
        default="copy,cortical,dmvfn",
        help="predictors the adaptive engine routes between (for ablations)",
    )
    learning = parser.add_argument_group("online learning")
    learning.add_argument(
        "--learn", action="store_true", help="keep training the cortical network on the stream"
    )
    learning.add_argument(
        "--from-scratch",
        action="store_true",
        help="start the cortical network untrained (implies --learn; no checkpoint needed)",
    )
    learning.add_argument("--learn-rate", type=float, default=2e-4)
    learning.add_argument(
        "--learn-steps", type=float, default=0.0, help="gradient steps per frame (0 = auto)"
    )
    learning.add_argument(
        "--anchor",
        type=float,
        default=1e-2,
        help="pull toward the starting weights; ignored with --from-scratch",
    )


def cortical_engine(args, mixed_precision: bool) -> CorticalEngine:
    if args.from_scratch:
        engine = CorticalEngine.fresh(
            device=args.device,
            width=args.width,
            height=args.height,
            mixed_precision=mixed_precision,
        )
    else:
        engine = CorticalEngine.from_checkpoint(
            args.checkpoint, device=args.device, mixed_precision=mixed_precision
        )
    if args.learn or args.from_scratch:
        # 0 = as fast as the duty cycle allows; a positive value caps steps per frame.
        steps = args.learn_steps or 10.0
        engine.enable_learning(
            learning_rate=args.learn_rate * (5.0 if args.from_scratch else 1.0),
            steps_per_frame=steps,
            anchor_weight=0.0 if args.from_scratch else args.anchor,
        )
    return engine


def create_engine(args):
    args.device = resolve_device(args.device)
    uses_cortical = args.engine == "cortical" or (
        args.engine == "adaptive" and "cortical" in args.experts
    )
    needs_cortical_checkpoint = uses_cortical and not args.from_scratch
    if (needs_cortical_checkpoint or args.engine == "dmvfn") and not args.checkpoint:
        raise SystemExit(f"--engine {args.engine} requires --checkpoint (or --from-scratch)")
    if (args.learn or args.from_scratch) and not uses_cortical:
        raise SystemExit("online learning needs the cortical network in the engine")
    if args.engine == "adaptive":
        names = [name.strip() for name in args.experts.split(",") if name.strip()]
        unknown = set(names) - {"copy", "cortical", "dmvfn", "flow"}
        if unknown or not names:
            raise SystemExit(f"--experts must name copy, cortical, dmvfn or flow; got {unknown}")
        if "dmvfn" in names and not args.dmvfn_checkpoint:
            raise SystemExit("--experts dmvfn requires --dmvfn-checkpoint")
        experts = {}
        if "cortical" in names:
            # Alternating these small models was faster in FP32 than BF16 on RTX 3090.
            experts["cortical"] = cortical_engine(args, mixed_precision=False)
        width = experts["cortical"].width if experts else args.width
        height = experts["cortical"].height if experts else args.height
        for name in names:
            if name == "copy":
                experts["copy"] = CopyEngine(width, height)
            elif name == "dmvfn":
                experts["dmvfn"] = DMVFNEngine(
                    args.dmvfn_checkpoint,
                    device=args.device,
                    width=width,
                    height=height,
                    mixed_precision=False,
                )
            elif name == "flow":
                experts["flow"] = OpticalFlowEngine(width=width, height=height)
        return AdaptiveEngine(
            {name: experts[name] for name in names},
            temperature=args.temperature,
            adaptation=args.adaptation,
            patch_size=args.patch_size,
        )
    if args.engine == "cortical":
        return cortical_engine(args, mixed_precision=not args.no_amp)
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


RELEASES = "https://github.com/froid1911/cortical-mirror/releases/download"
DEMO_ASSETS = {
    "cortical.pt": f"{RELEASES}/v0.2.0/cortical-mirror-v0.2.0.pt",
    "dmvfn_vimeo.pkl": f"{RELEASES}/v0.2.0/dmvfn_vimeo.pkl",
    "sample.mp4": f"{RELEASES}/v0.2.0/sample.mp4",
    # The learning cortex's adults, one per phase, each trained on hours of UCF101.
    **{f"cortex-adult-p{p}.pt": f"{RELEASES}/v0.3.0/cortex-adult-p{p}.pt" for p in (1, 2, 3)},
    # The scene graph's adult: the persistence it learned from occlusion scenes.
    "scene-adult.pt": f"{RELEASES}/v0.4.0/scene-adult.pt",
    # The motion cells MT learned from video (v0.7), with the V1 they read.
    "mt-adult.pt": f"{RELEASES}/v0.9.0/mt-adult.pt",
}


def fetch(name: str, folder: Path) -> Path:
    path = folder / name
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name} to {folder}", file=sys.stderr, flush=True)
        temporary = path.with_suffix(path.suffix + ".part")
        try:
            with (
                urllib.request.urlopen(DEMO_ASSETS[name]) as response,
                temporary.open("wb") as out,
            ):
                shutil.copyfileobj(response, out)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            if name == "sample.mp4":
                alternative = "pass --source with a video path"
            elif name.startswith("cortex-adult"):
                alternative = "pass --adult with a checkpoint you have"
            elif name == "scene-adult.pt":
                alternative = "pass --newborn, or --adult with a scene checkpoint"
            elif name == "mt-adult.pt":
                alternative = "pass --motion flow, or --mt with a motion checkpoint"
            else:
                alternative = "leave out --pretrained to start untrained"
            raise SystemExit(
                f"could not download {name} from {DEMO_ASSETS[name]} ({error}). "
                f"Place the file at {path} yourself, or {alternative}."
            ) from error
        temporary.replace(path)
    return path


def camera_available(source: str) -> bool:
    if "://" in source:
        return True  # a network stream; the server reports if it cannot be opened
    if not source.isdigit():
        return Path(source).exists()
    capture = cv2.VideoCapture(int(source))
    try:
        return capture.isOpened() and capture.read()[0]
    finally:
        capture.release()


def demo_arguments(args) -> None:
    """Fill in everything `cortical-mirror demo` needs, downloading on first use.

    By default the hierarchy starts untrained and learns from the stream, routed
    only against persistence, so everything the mirror shows was learned on the
    spot. `--pretrained` loads the trained weights and, on CUDA, adds DMVFN.
    """

    folder = args.assets
    args.device = resolve_device(args.device)
    args.engine = "adaptive"
    args.from_scratch = not args.pretrained
    args.learn = True
    if args.from_scratch:
        args.experts = "copy,cortical"
        if args.device == "cpu":
            # Measured on a 12-thread laptop CPU: about 1 learning step per second,
            # a tenth of the GPU rate (reports/latency-laptop-cpu.json).
            print(
                "no CUDA device: learning from scratch on the CPU takes minutes, not seconds",
                file=sys.stderr,
            )
    else:
        args.checkpoint = fetch("cortical.pt", folder)
        if args.device == "cpu":
            # DMVFN would hold a laptop CPU near 10 fps, and a CPU learner cannot keep up.
            args.experts = "copy,cortical"
            args.learn = False
            print(
                "no CUDA device: routing persistence and the trained hierarchy; "
                "live learning is off",
                file=sys.stderr,
            )
        else:
            args.dmvfn_checkpoint = fetch("dmvfn_vimeo.pkl", folder)
    if not camera_available(args.source):
        print(f"no camera at {args.source!r}; playing the sample clip", file=sys.stderr)
        args.source = str(fetch("sample.mp4", folder))


def twins_arguments(args) -> None:
    """Fill in what `cortical-mirror twins` needs: the adult of the chosen phase,
    downloaded on first use unless --adult names one, and the sample clip if
    there is no camera."""

    if args.adult is None:
        args.adult = fetch(f"cortex-adult-p{args.phase}.pt", args.assets)
    if not camera_available(args.source):
        print(f"no camera at {args.source!r}; playing the sample clip", file=sys.stderr)
        args.source = str(fetch("sample.mp4", args.assets))


def twins_engine(args):
    """Newborn (fresh, learning) beside the saved adult, for `cortical-mirror twins`."""

    from .cortex.engine import CortexEngine
    from .cortex.twin import TwinEngine

    device = resolve_device(args.device)
    adult = CortexEngine.load(args.adult, device=device, learn=not args.frozen_adult)
    newborn = CortexEngine(device=device, learn=True, phase=adult.phase)
    return TwinEngine(newborn, adult)


def scene_arguments(args) -> None:
    if args.adult is None and not args.newborn:
        args.adult = fetch("scene-adult.pt", args.assets)
    learned = getattr(args, "motion", "flow") == "learned" and args.source != "world"
    if learned and getattr(args, "mt", None) is None:
        args.mt = fetch("mt-adult.pt", args.assets)


def scene_graph(args):
    from .scene.graph import NEWBORN_PERSISTENCE, SceneGraph

    if args.newborn:
        return SceneGraph(persistence=NEWBORN_PERSISTENCE)
    return SceneGraph.load(args.adult)


def main() -> None:
    parser = argparse.ArgumentParser(prog="cortical-mirror", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser(
        "demo", help="one command: open the camera and learn to predict it from scratch"
    )
    demo.set_defaults(engine="adaptive")
    demo.add_argument("--source", default="0", help="camera index or video path")
    demo.add_argument("--host", default="127.0.0.1")
    demo.add_argument("--port", type=int, default=8080)
    demo.add_argument("--assets", type=Path, default=Path.home() / ".cache" / "cortical-mirror")
    demo.add_argument(
        "--pretrained",
        action="store_true",
        help="start from the trained weights (and add DMVFN on CUDA) instead of from scratch",
    )
    demo.add_argument("--horizon", type=int, default=1, help="frames ahead to show (1-8)")
    demo.add_argument("--device", default="auto")

    live = commands.add_parser("serve", help="serve the live mirror in a browser")
    add_engine_arguments(live)
    live.add_argument("--source", default="0", help="camera index or video path")
    live.add_argument("--host", default="127.0.0.1")
    live.add_argument("--port", type=int, default=8080)
    live.add_argument("--no-loop", action="store_true", help="stop at the end of a video source")
    live.add_argument("--max-frames", type=int, default=0)
    live.add_argument(
        "--horizon", type=int, default=1, help="frames ahead to show in the mirror (1-8)"
    )

    twins = commands.add_parser(
        "twins", help="the learning cortex: a newborn beside an adult, both watching you"
    )
    twins.add_argument(
        "--adult", type=Path, help="adult cortex checkpoint (default: download the --phase adult)"
    )
    twins.add_argument(
        "--phase", type=int, choices=(1, 2, 3), default=1, help="which adult to download"
    )
    twins.add_argument("--assets", type=Path, default=Path.home() / ".cache" / "cortical-mirror")
    twins.add_argument("--frozen-adult", action="store_true", help="the adult stops learning")
    twins.add_argument("--source", default="0", help="camera index, video path or stream URL")
    twins.add_argument("--host", default="127.0.0.1")
    twins.add_argument("--port", type=int, default=8080)
    twins.add_argument("--device", default="auto")
    twins.add_argument("--no-loop", action="store_true")
    twins.add_argument("--max-frames", type=int, default=0)

    scene = commands.add_parser(
        "scene", help="the world explained as a scene graph you can edit (v0.4)"
    )
    scene.add_argument(
        "--source", default="world", help="'world' (the sprite world), a camera index or a video"
    )
    scene.add_argument("--adult", type=Path, help="scene adult (default: download it)")
    scene.add_argument("--newborn", action="store_true", help="forget what it cannot see")
    scene.add_argument(
        "--motion",
        choices=("learned", "flow"),
        default="learned",
        help="for a camera or video: motion from learned MT cells (v0.7) or optical flow (v0.6)",
    )
    scene.add_argument("--mt", type=Path, help="MT adult (default: download it)")
    scene.add_argument("--assets", type=Path, default=Path.home() / ".cache" / "cortical-mirror")
    scene.add_argument(
        "--device",
        default="auto",
        help="where the graph searches: cpu, cuda (auto: cuda if present)",
    )
    scene.add_argument("--host", default="127.0.0.1")
    scene.add_argument("--port", type=int, default=8080)

    offline = commands.add_parser(
        "evaluate", help="evaluate predictions against their exact target frames"
    )
    add_engine_arguments(offline)
    offline.add_argument("--source", required=True, help="video path")
    offline.add_argument("--output", type=Path, required=True)
    offline.add_argument("--max-frames", type=int, default=120)
    offline.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()

    if args.command == "scene":
        from .scene.live import SceneLive, SpriteSource, VideoSource
        from .scene.live import serve as serve_scene

        scene_arguments(args)
        import os

        import torch

        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # The array search is the faster one on a CPU.
        os.environ["SCENE_DEVICE"] = "" if device == "cpu" else device
        from .scene.front import NormalisedFront
        from .scene.video import VideoGraph

        if args.source == "world":
            state = SceneLive(SpriteSource(), scene_graph(args))
        else:
            # A camera or a video: register the camera's own motion, and match in the
            # normalised code, which ignores changes of exposure.
            persistence = scene_graph(args).persistence
            if args.motion == "learned":
                from .cortex.mt import MotionCortex

                video = VideoGraph(
                    front=NormalisedFront(),
                    persistence=persistence,
                    motion_source="learned",
                    mt=MotionCortex.load(args.mt),
                )
            else:
                video = VideoGraph(front=NormalisedFront(), persistence=persistence)
            state = SceneLive(VideoSource(args.source), video=video)
        serve_scene(state, args.host, args.port)
        return
    if args.command == "twins":
        twins_arguments(args)
        serve(
            twins_engine(args),
            source=args.source,
            engine_name="twins",
            host=args.host,
            port=args.port,
            loop=not args.no_loop,
            max_frames=args.max_frames,
        )
        return
    if args.command == "demo":
        defaults = parser.parse_args(["serve", "--engine", "adaptive"])
        args = argparse.Namespace(**(vars(defaults) | vars(args)))
        args.loop, args.no_loop, args.max_frames = True, False, 0
        demo_arguments(args)
    engine = create_engine(args)
    if args.command in ("serve", "demo"):
        serve(
            engine,
            source=args.source,
            engine_name=args.engine,
            host=args.host,
            port=args.port,
            loop=not args.no_loop,
            max_frames=args.max_frames,
            horizon=args.horizon,
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
