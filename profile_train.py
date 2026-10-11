"""
Short training profiler for finding throughput bottlenecks.

It times the expensive parts of a few training batches:
data loading, CPU-to-GPU transfer, forward pass, loss, backward, and optimizer.
Run this on Kaggle before long training runs to see what is actually slow.
"""

import argparse
import random
import time
from collections import defaultdict

import torch
import torch.nn as nn
from tqdm import tqdm

from config import cfg
from dataset import get_dataloaders
from loss import BCEDiceLoss
from model import ResNetUNet


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def elapsed_stage(timers, name, device, fn):
    sync_if_cuda(device)
    start = time.perf_counter()
    result = fn()
    sync_if_cuda(device)
    timers[name] += time.perf_counter() - start
    return result


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def set_encoder_freeze(model, freeze):
    for name, param in unwrap(model).named_parameters():
        if name.startswith("encoder."):
            param.requires_grad = not freeze


def summarize(title, timers, steps):
    total = sum(timers.values())
    print(f"\n{title}")
    print("-" * len(title))
    print(f"{'stage':<16} {'total_s':>10} {'ms/batch':>12} {'percent':>9}")
    for name, value in sorted(timers.items(), key=lambda item: item[1], reverse=True):
        pct = (value / total * 100.0) if total > 0 else 0.0
        print(f"{name:<16} {value:10.3f} {value / max(steps, 1) * 1000.0:12.1f} {pct:8.1f}%")
    print(f"{'measured_total':<16} {total:10.3f} {total / max(steps, 1) * 1000.0:12.1f} {100.0:8.1f}%")


def profile_raw_dataset(loader, samples):
    if samples <= 0:
        return

    dataset = loader.dataset
    rng = random.Random(cfg.seed)
    count = min(samples, len(dataset))
    timings = []

    print(f"\nProfiling direct dataset __getitem__ for {count} random samples...")
    for _ in tqdm(range(count), desc="Dataset samples", leave=False):
        idx = rng.randrange(len(dataset))
        start = time.perf_counter()
        _ = dataset[idx]
        timings.append(time.perf_counter() - start)

    timings.sort()
    avg_ms = sum(timings) / len(timings) * 1000.0
    p50_ms = timings[len(timings) // 2] * 1000.0
    p95_ms = timings[int(len(timings) * 0.95) - 1] * 1000.0
    print(f"dataset_getitem_ms: avg={avg_ms:.1f}, p50={p50_ms:.1f}, p95={p95_ms:.1f}")


def profile_train_batches(model, loader, criterion, optimizer, scaler, device, use_amp, max_batches):
    model.train()
    timers = defaultdict(float)
    steps = 0

    iterator_end = time.perf_counter()
    pbar = tqdm(loader, total=min(max_batches, len(loader)), desc="Profile train", leave=False)
    for images, masks in pbar:
        timers["data_wait"] += time.perf_counter() - iterator_end

        images, masks = elapsed_stage(
            timers,
            "to_device",
            device,
            lambda: (images.to(device, non_blocking=True), masks.to(device, non_blocking=True)),
        )

        elapsed_stage(timers, "zero_grad", device, lambda: optimizer.zero_grad(set_to_none=True))

        def forward_pass():
            with torch.autocast(device_type=device.type, enabled=use_amp):
                return model(images)

        logits = elapsed_stage(timers, "forward", device, forward_pass)

        def loss_pass():
            with torch.autocast(device_type=device.type, enabled=use_amp):
                return criterion(logits.float(), masks)

        loss = elapsed_stage(timers, "loss", device, loss_pass)

        def backward_pass():
            scaler.scale(loss).backward()
            if cfg.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)

        elapsed_stage(timers, "backward", device, backward_pass)

        def optimizer_step():
            scaler.step(optimizer)
            scaler.update()

        elapsed_stage(timers, "optimizer", device, optimizer_step)

        steps += 1
        iterator_end = time.perf_counter()
        if steps >= max_batches:
            break

    summarize("Training Batch Timing", timers, steps)
    return timers


@torch.no_grad()
def profile_val_batches(model, loader, criterion, device, use_amp, max_batches):
    model.eval()
    timers = defaultdict(float)
    steps = 0

    iterator_end = time.perf_counter()
    pbar = tqdm(loader, total=min(max_batches, len(loader)), desc="Profile val", leave=False)
    for images, masks in pbar:
        timers["data_wait"] += time.perf_counter() - iterator_end

        images, masks = elapsed_stage(
            timers,
            "to_device",
            device,
            lambda: (images.to(device, non_blocking=True), masks.to(device, non_blocking=True)),
        )

        def forward_pass():
            with torch.autocast(device_type=device.type, enabled=use_amp):
                return model(images)

        logits = elapsed_stage(timers, "forward", device, forward_pass)
        _ = elapsed_stage(timers, "loss", device, lambda: criterion(logits.float(), masks))

        steps += 1
        iterator_end = time.perf_counter()
        if steps >= max_batches:
            break

    summarize("Validation Batch Timing", timers, steps)
    return timers


def parse_args():
    parser = argparse.ArgumentParser(description="Profile a few training batches.")
    parser.add_argument("--data-dir", default=None, help="Override cfg.data_dir")
    parser.add_argument("--batch-size", type=int, default=None, help="Override cfg.batch_size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override cfg.num_workers")
    parser.add_argument("--image-size", type=int, default=None, help="Override cfg.image_size")
    parser.add_argument("--train-batches", type=int, default=50)
    parser.add_argument("--val-batches", type=int, default=10)
    parser.add_argument("--dataset-samples", type=int, default=64)
    parser.add_argument("--single-gpu", action="store_true", help="Disable nn.DataParallel for comparison")
    parser.add_argument("--no-amp", action="store_true", help="Disable autocast/GradScaler")
    parser.add_argument("--freeze-encoder", action="store_true", help="Profile frozen encoder behavior")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.image_size is not None:
        cfg.image_size = args.image_size

    cfg.display()
    device = torch.device(cfg.device)
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print("\nBuilding DataLoaders...")
    loader_start = time.perf_counter()
    train_loader, val_loader = get_dataloaders(cfg)
    print(f"DataLoader setup time: {time.perf_counter() - loader_start:.2f}s")

    profile_raw_dataset(train_loader, args.dataset_samples)

    print("\nBuilding model...")
    model = ResNetUNet(pretrained=True).to(device)
    if args.freeze_encoder:
        set_encoder_freeze(model, True)

    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    if num_gpus > 1 and not args.single_gpu:
        print(f"Using nn.DataParallel on {num_gpus} GPUs")
        model = nn.DataParallel(model)
    elif num_gpus > 1:
        print("Using one GPU only for comparison")

    criterion = BCEDiceLoss()
    enc_params = [p for n, p in unwrap(model).named_parameters() if n.startswith("encoder.")]
    dec_params = [p for n, p in unwrap(model).named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.Adam(
        [
            {"params": enc_params, "lr": cfg.encoder_lr},
            {"params": dec_params, "lr": cfg.lr},
        ]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(
        f"\nProfiling {args.train_batches} train batches and {args.val_batches} val batches "
        f"(AMP={'on' if use_amp else 'off'})..."
    )
    profile_train_batches(
        model,
        train_loader,
        criterion,
        optimizer,
        scaler,
        device,
        use_amp,
        max_batches=args.train_batches,
    )
    if args.val_batches > 0:
        profile_val_batches(
            model,
            val_loader,
            criterion,
            device,
            use_amp,
            max_batches=args.val_batches,
        )

    print("\nHow to read this:")
    print("- high data_wait: DataLoader/image preprocessing/disk is starving the GPU")
    print("- high to_device: CPU-to-GPU transfer or pinned memory issue")
    print("- high forward/backward: model compute is the real bottleneck")
    print("- compare --single-gpu vs default to see whether DataParallel helps")


if __name__ == "__main__":
    main()
