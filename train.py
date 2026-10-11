"""
Clean, straightforward training script for Human Background Removal.
Supports Single-GPU, Multi-GPU (nn.DataParallel), and CPU with tqdm progress tracking.
"""

import csv
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from config import cfg
from dataset import get_dataloaders
from loss import BCEDiceLoss, dice_score, iou_score
from model import ResNetUNet


def unwrap(model: nn.Module) -> nn.Module:
    """Unwraps model from DataParallel wrapper if present."""
    return model.module if isinstance(model, nn.DataParallel) else model


def set_encoder_freeze(model: nn.Module, freeze: bool):
    """Freezes or unfreezes encoder parameters."""
    for name, param in unwrap(model).named_parameters():
        if name.startswith("encoder."):
            param.requires_grad = not freeze


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, epoch: int):
    model.train()
    total_loss, total_samples = 0.0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:2d}/{cfg.epochs} [Train]", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits.float(), masks)

        scaler.scale(loss).backward()
        if cfg.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        b = images.size(0)
        total_loss += loss.item() * b
        total_samples += b

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg_loss": f"{total_loss / total_samples:.4f}",
        })

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp, epoch: int):
    model.eval()
    total_loss, total_dice, total_iou, total_samples = 0.0, 0.0, 0.0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:2d}/{cfg.epochs} [Val]  ", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
        logits = logits.float()

        b = images.size(0)
        total_loss += criterion(logits, masks).item() * b
        total_dice += dice_score(logits, masks) * b
        total_iou += iou_score(logits, masks) * b
        total_samples += b

        pbar.set_postfix({
            "val_loss": f"{total_loss / total_samples:.4f}",
            "dice": f"{total_dice / total_samples:.4f}",
            "iou": f"{total_iou / total_samples:.4f}",
        })

    return (
        total_loss / total_samples,
        total_dice / total_samples,
        total_iou / total_samples,
    )


def main():
    cfg.display()
    device = torch.device(cfg.device)
    use_amp = device.type == "cuda"
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 1. DataLoaders
    train_loader, val_loader = get_dataloaders(cfg)

    # 2. Model & Optimization
    model = ResNetUNet(pretrained=True).to(device)
    criterion = BCEDiceLoss()

    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    dec_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.Adam(
        [
            {"params": enc_params, "lr": cfg.encoder_lr},
            {"params": dec_params, "lr": cfg.lr},
        ]
    )

    # Multi-GPU DataParallel Support (e.g. 2x T4 on Kaggle)
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"--> Multi-GPU detected: using {num_gpus} GPUs with nn.DataParallel")
        model = nn.DataParallel(model)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 3. Training Loop & History Logging
    hist_path = os.path.join(cfg.output_dir, "history.csv")
    best_path = os.path.join(cfg.output_dir, "best_model.pth")
    last_path = os.path.join(cfg.output_dir, "last_model.pth")

    with open(hist_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "lr", "time_s"]
        )
        writer.writeheader()

    best_dice = 0.0
    no_improve = 0

    print("\nStarting Training...")
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        # Warmup: freeze encoder for first N epochs
        is_frozen = epoch <= cfg.freeze_epochs
        set_encoder_freeze(model, freeze=is_frozen)

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp, epoch=epoch
        )
        val_loss, val_dice, val_iou = evaluate(
            model, val_loader, criterion, device, use_amp, epoch=epoch
        )
        scheduler.step()

        elapsed = time.time() - t0
        curr_lr = optimizer.param_groups[1]["lr"]

        # Log metrics
        with open(hist_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "lr", "time_s"]
            )
            writer.writerow({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_dice": round(val_dice, 4),
                "val_iou": round(val_iou, 4),
                "lr": round(curr_lr, 7),
                "time_s": round(elapsed, 1),
            })

        status_msg = (
            f"Epoch {epoch:2d}/{cfg.epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f} | LR: {curr_lr:.2e} | {elapsed:.1f}s"
            f"{' [FROZEN]' if is_frozen else ''}"
        )
        print(status_msg)

        # Checkpoint: Save best model (unwrapped state dict for seamless single-GPU/CPU inference)
        if val_dice > best_dice:
            best_dice = val_dice
            no_improve = 0
            torch.save(unwrap(model).state_dict(), best_path)
            print(f"  --> Saved new best model (Dice: {best_dice:.4f})")
        else:
            no_improve += 1

        torch.save(unwrap(model).state_dict(), last_path)

        # Early Stopping
        if no_improve >= cfg.patience:
            print(f"\nEarly stopping triggered: No val Dice improvement for {cfg.patience} epochs.")
            break

    print(f"\nTraining Complete! Best Validation Dice: {best_dice:.4f}")
    print(f"Weights saved at: {best_path}")


if __name__ == "__main__":
    main()
