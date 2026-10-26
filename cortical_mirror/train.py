"""Self-supervised training for the causal cortical predictor."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import ClipConfig, SyntheticMotionDataset, VideoClipDataset, split_videos
from .engines import load_checkpoint
from .model import CorticalFlowNet


def seed_worker(_: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool, seed: int, cuda: bool):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=cuda,
        drop_last=shuffle,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def flow_error(predicted: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Mean endpoint error in full-resolution pixels."""

    # The epsilon keeps the gradient finite where prediction and teacher agree exactly.
    return ((predicted.float() - teacher).square().sum(dim=1) + 1e-6).sqrt().mean()


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    scheduler=None,
    max_steps: int = 0,
    mixed_precision=True,
    teacher_weight: float = 0.0,
    burn_in: int = 0,
):
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "prediction_mae": 0.0,
        "copy_mae": 0.0,
        "wins": 0,
        "transitions": 0,
        "flow_epe": 0.0,
        "zero_flow_epe": 0.0,
        "flow_speed": 0.0,
        "teacher_speed": 0.0,
        "flow_transitions": 0,
    }
    clips = 0
    start = time.perf_counter()
    progress = tqdm(loader, leave=False, desc="train" if training else "validate")
    for batch_index, batch in enumerate(progress):
        if max_steps and batch_index >= max_steps:
            break
        frames, teacher = batch if isinstance(batch, (list, tuple)) else (batch, None)
        frames = frames.to(device, non_blocking=True)
        if teacher is not None:
            teacher = teacher.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        state = None
        state_frame = None
        if burn_in:
            # Settle the recurrent state on earlier frames first, so the network is
            # trained in the steady state a live camera puts it in, not only from zero.
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=mixed_precision and device.type == "cuda",
                ),
            ):
                for index in range(burn_in):
                    state = model(frames[:, index], state).state.detached()
            state_frame = frames[:, burn_in - 1]
            frames = frames[:, burn_in:]
        losses = []
        batch_prediction = 0.0
        batch_copy = 0.0
        batch_wins = 0
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=mixed_precision and device.type == "cuda",
            ):
                for index in range(frames.shape[1]):
                    output = model(frames[:, index], state)
                    if output.has_ground_truth:
                        loss = model.loss(output)
                        if teacher is not None:
                            # Teacher flow j is the clip's transition from frame j to j + 1.
                            target = teacher[:, burn_in + index - 1]
                            epe = flow_error(output.previous_flow, target)
                            if training and teacher_weight > 0:
                                loss = loss + teacher_weight * epe
                            totals["flow_epe"] += float(epe)
                            totals["zero_flow_epe"] += float(
                                torch.linalg.vector_norm(target, dim=1).mean()
                            )
                            totals["flow_speed"] += float(
                                torch.linalg.vector_norm(output.previous_flow.float(), dim=1).mean()
                            )
                            totals["teacher_speed"] += float(
                                torch.linalg.vector_norm(target, dim=1).mean()
                            )
                            totals["flow_transitions"] += 1
                        losses.append(loss)
                        prediction_mae = output.observed_error.mean()
                        previous = frames[:, index - 1] if index else state_frame
                        copy_mae = (frames[:, index] - previous).abs().mean()
                        batch_prediction += float(prediction_mae)
                        batch_copy += float(copy_mae)
                        batch_wins += int(prediction_mae < copy_mae)
                    state = output.state if training else output.state.detached()
            if training:
                loss = torch.stack(losses).mean()
                loss.backward()
                norm = clip_grad_norm_(model.parameters(), 1.0)
                if torch.isfinite(norm):
                    optimizer.step()
                else:
                    totals["skipped"] = totals.get("skipped", 0) + 1
                if scheduler is not None:
                    scheduler.step()
        transitions = len(losses)
        if not transitions:
            continue
        clips += 1
        totals["loss"] += float(torch.stack(losses).mean().detach())
        totals["prediction_mae"] += batch_prediction
        totals["copy_mae"] += batch_copy
        totals["wins"] += batch_wins
        totals["transitions"] += transitions
        progress.set_postfix(
            loss=f"{totals['loss'] / clips:.4f}",
            mae=f"{totals['prediction_mae'] / totals['transitions']:.4f}",
            copy=f"{totals['copy_mae'] / totals['transitions']:.4f}",
        )
    if not clips:
        raise RuntimeError("no complete clips were processed")
    metrics = {
        "loss": totals["loss"] / clips,
        "prediction_mae": totals["prediction_mae"] / totals["transitions"],
        "copy_mae": totals["copy_mae"] / totals["transitions"],
        "beat_copy": totals["wins"] / totals["transitions"],
        "clips": clips,
        "fps": clips / max(time.perf_counter() - start, 1e-6),
        "skipped_steps": totals.get("skipped", 0),
    }
    if totals["flow_transitions"]:
        count = totals["flow_transitions"]
        # A model that predicts no motion scores zero_flow_epe; lower is better.
        metrics |= {
            key: totals[key] / count
            for key in ("flow_epe", "zero_flow_epe", "flow_speed", "teacher_speed")
        }
    return metrics


def save_checkpoint(
    path: Path, model, optimizer, epoch: int, metrics: dict, args, scheduler=None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_config": model.checkpoint_config(),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "image_size": [args.width, args.height],
        "epoch": epoch,
        "metrics": metrics,
        "train_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="folder containing training videos")
    parser.add_argument("--synthetic", type=int, default=0, help="use N synthetic smoke-test clips")
    parser.add_argument("--output", type=Path, default=Path("runs/cortical"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument(
        "--burn-in", type=int, default=0, help="frames run without gradient before each clip"
    )
    parser.add_argument("--clips-per-video", type=int, default=4)
    parser.add_argument("--validation-videos", type=int, default=0, help="0 uses all of them")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--channels", default="16,32,64")
    parser.add_argument("--strides", default="1,4,8")
    parser.add_argument("--cell-limit", type=float, default=4.0, help="0 leaves cells unbounded")
    parser.add_argument(
        "--flow-teacher",
        type=float,
        default=0.0,
        help="initial weight of classical-flow distillation; decays linearly to zero",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-steps", type=int, default=0, help="cap clips per epoch for smoke tests"
    )
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data and not args.synthetic:
        raise SystemExit("choose --data or --synthetic")
    if args.frames < 3:
        raise SystemExit("--frames must be at least 3")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    config = ClipConfig(
        width=args.width,
        height=args.height,
        frames=args.burn_in + args.frames,
        clips_per_video=args.clips_per_video,
    )
    if args.data:
        split = split_videos(args.data)
        validation_videos = split.validation
        if args.validation_videos:
            step = max(1, len(validation_videos) // args.validation_videos)
            validation_videos = validation_videos[::step][: args.validation_videos]
        train_set = VideoClipDataset(split.train, config, with_flow=args.flow_teacher > 0)
        validation_config = ClipConfig(
            width=args.width,
            height=args.height,
            frames=args.burn_in + args.frames,
            clips_per_video=1,
        )
        validation_set = VideoClipDataset(
            validation_videos, validation_config, deterministic=True, with_flow=True
        )
        # The test split is never loaded here: it is reserved for tools/benchmark.py.
        print(
            f"videos: train={len(split.train)} validation={len(validation_videos)} "
            f"test(reserved)={len(split.test)}"
        )
    else:
        train_set = SyntheticMotionDataset(args.synthetic, config)
        validation_set = SyntheticMotionDataset(max(16, args.synthetic // 10), config)
    train_loader = make_loader(
        train_set, args.batch_size, args.workers, True, args.seed, device.type == "cuda"
    )
    validation_loader = make_loader(
        validation_set, args.batch_size, args.workers, False, args.seed + 1, device.type == "cuda"
    )

    channels = tuple(int(value) for value in args.channels.split(","))
    strides = tuple(int(value) for value in args.strides.split(","))
    model = CorticalFlowNet(channels=channels, strides=strides, cell_limit=args.cell_limit)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = (
        min(len(train_loader), args.max_steps) if args.max_steps else len(train_loader)
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=max(1, steps_per_epoch * args.epochs),
        pct_start=0.05,
    )
    start_epoch = 0
    best_validation = float("inf")
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_validation = checkpoint["metrics"].get("validation_mae", float("inf"))
        print(f"resumed {args.resume} at epoch {start_epoch}")

    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, args.epochs):
        teacher_weight = args.flow_teacher * max(0.0, 1.0 - epoch / max(1, args.epochs - 1))
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
            max_steps=args.max_steps,
            mixed_precision=not args.no_amp,
            teacher_weight=teacher_weight,
            burn_in=args.burn_in,
        )
        with torch.no_grad():
            validation_metrics = run_epoch(
                model,
                validation_loader,
                device,
                max_steps=args.max_steps,
                mixed_precision=not args.no_amp,
                burn_in=args.burn_in,
            )
        record = {
            "epoch": epoch,
            "teacher_weight": teacher_weight,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        print(json.dumps(record))
        with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        # Select on validation pixel error only; the teacher term is a training aid.
        record["validation_mae"] = validation_metrics["prediction_mae"]
        save_checkpoint(args.output / "last.pt", model, optimizer, epoch, record, args, scheduler)
        if validation_metrics["prediction_mae"] < best_validation:
            best_validation = validation_metrics["prediction_mae"]
            save_checkpoint(
                args.output / "best.pt", model, optimizer, epoch, record, args, scheduler
            )


if __name__ == "__main__":
    main()
