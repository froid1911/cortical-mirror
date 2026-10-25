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


def run_epoch(model, loader, device, optimizer=None, max_steps: int = 0, mixed_precision=True):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "prediction_mae": 0.0, "copy_mae": 0.0, "wins": 0, "transitions": 0}
    clips = 0
    start = time.perf_counter()
    progress = tqdm(loader, leave=False, desc="train" if training else "validate")
    for batch_index, frames in enumerate(progress):
        if max_steps and batch_index >= max_steps:
            break
        frames = frames.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        state = None
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
                        losses.append(model.loss(output))
                        prediction_mae = output.observed_error.mean()
                        copy_mae = (frames[:, index] - frames[:, index - 1]).abs().mean()
                        batch_prediction += float(prediction_mae)
                        batch_copy += float(copy_mae)
                        batch_wins += int(prediction_mae < copy_mae)
                    state = output.state if training else output.state.detached()
            if training:
                loss = torch.stack(losses).mean()
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
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
    return {
        "loss": totals["loss"] / clips,
        "prediction_mae": totals["prediction_mae"] / totals["transitions"],
        "copy_mae": totals["copy_mae"] / totals["transitions"],
        "beat_copy": totals["wins"] / totals["transitions"],
        "clips": clips,
        "fps": clips / max(time.perf_counter() - start, 1e-6),
    }


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_config": model.checkpoint_config(),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
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
    parser.add_argument("--clips-per-video", type=int, default=4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--channels", default="24,48,96")
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
        frames=args.frames,
        clips_per_video=args.clips_per_video,
    )
    if args.data:
        train_videos, validation_videos = split_videos(args.data)
        train_set = VideoClipDataset(train_videos, config)
        validation_config = ClipConfig(
            width=args.width, height=args.height, frames=args.frames, clips_per_video=1
        )
        validation_set = VideoClipDataset(validation_videos, validation_config, deterministic=True)
        print(f"videos: train={len(train_videos)} validation={len(validation_videos)}")
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
    model = CorticalFlowNet(channels=channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start_epoch = 0
    best_validation = float("inf")
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_validation = checkpoint["metrics"].get("validation_loss", float("inf"))
        print(f"resumed {args.resume} at epoch {start_epoch}")

    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            max_steps=args.max_steps,
            mixed_precision=not args.no_amp,
        )
        with torch.no_grad():
            validation_metrics = run_epoch(
                model,
                validation_loader,
                device,
                max_steps=args.max_steps,
                mixed_precision=not args.no_amp,
            )
        record = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        print(json.dumps(record))
        with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        record["validation_loss"] = validation_metrics["loss"]
        save_checkpoint(args.output / "last.pt", model, optimizer, epoch, record, args)
        if validation_metrics["loss"] < best_validation:
            best_validation = validation_metrics["loss"]
            save_checkpoint(args.output / "best.pt", model, optimizer, epoch, record, args)


if __name__ == "__main__":
    main()
