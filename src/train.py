import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.dataset.pets import get_dataloaders
from src.losses import BCEDiceLoss
from src.metrics import dice_score, iou_score
from src.models.unet import ResNetUnet

REQUIRED_KEYS = {
    "seed": int,
    "data_root": str,
    "val_fraction": float,
    "image_size": int,
    "num_workers": int,
    "batch_size": int,
    "epochs": int,
    "lr": float,
    "encoder_lr": float,
    "loss": str,
    "encoder": str,
    "pretrained": bool,
    "output_dir": str,
}

OPTIONAL_DEFAULTS = {
    "freeze_epochs": 0,       # Initial epochs to keep pretrained encoder frozen
    "max_grad_norm": 1.0,     # Gradient clipping max norm (<= 0 to disable)
    "min_lr": 1.0e-6,         # Minimum LR for CosineAnnealing scheduler
    "patience": 10,           # Early stopping patience
    "scheduler": "cosine",    # LR scheduler type ('cosine' or 'none')
}


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Check required keys
    for key, typ in REQUIRED_KEYS.items():
        if key not in cfg or cfg[key] is None:
            raise ValueError(f"Missing config key: {key}")
        value = cfg[key]
        if typ is float:
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        else:
            ok = isinstance(value, typ)
        if not ok:
            raise TypeError(
                f"Config key '{key}' should be {typ.__name__}, got "
                f"{type(value).__name__} ({value!r}). "
                f"Hint: write 1.0e-3, not 1e-3, for floats in yaml."
            )

    # Apply optional defaults
    for opt_key, default_val in OPTIONAL_DEFAULTS.items():
        if opt_key not in cfg or cfg[opt_key] is None:
            cfg[opt_key] = default_val

    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap(model):
    """Return the underlying model if it is wrapped in DataParallel."""
    return model.module if isinstance(model, nn.DataParallel) else model


def set_encoder_freeze(model, freeze: bool):
    """Freezes or unfreezes all parameters belonging to the encoder."""
    unwrapped = unwrap(model)
    for name, param in unwrapped.named_parameters():
        if name.startswith("encoder."):
            param.requires_grad = not freeze


def build_loss(name):
    losses = {"bce_dice": BCEDiceLoss}
    if name not in losses:
        raise ValueError(f"Unknown loss '{name}'. Options: {list(losses)}")
    return losses[name]()


def build_optimizer(model, cfg):
    # Pretrained encoder gets a smaller learning rate than the new decoder + head
    unwrapped = unwrap(model)
    enc = [p for n, p in unwrapped.named_parameters() if n.startswith("encoder.")]
    rest = [p for n, p in unwrapped.named_parameters() if not n.startswith("encoder.")]
    return torch.optim.Adam(
        [
            {"params": enc, "lr": cfg["encoder_lr"]},
            {"params": rest, "lr": cfg["lr"]},
        ]
    )


def build_scheduler(optimizer, cfg):
    sched_type = cfg.get("scheduler", "cosine").lower()
    if sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"], eta_min=cfg["min_lr"]
        )
    elif sched_type == "none":
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {sched_type}. Use 'cosine' or 'none'.")


def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, use_amp, max_grad_norm=1.0
):
    model.train()
    total, n = 0.0, 0
    for images, masks, valid in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
        # Loss computed in float32 to prevent fp16 overflow over pixel sums
        loss = criterion(logits.float(), masks, valid)

        scaler.scale(loss).backward()

        # Gradient clipping with AMP
        if max_grad_norm is not None and max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        total += loss.item() * images.size(0)
        n += images.size(0)
    return total / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    loss_sum = dice_sum = iou_sum = 0.0
    n = 0
    for images, masks, valid in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
        logits = logits.float()

        b = images.size(0)
        loss_sum += criterion(logits, masks, valid).item() * b
        dice_sum += dice_score(logits, masks, valid).item() * b
        iou_sum += iou_score(logits, masks, valid).item() * b
        n += b
    return loss_sum / n, dice_sum / n, iou_sum / n


def save_checkpoint(
    path, model, optimizer, scheduler, scaler, epoch, best_dice, no_improve
):
    torch.save(
        {
            "model": unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_dice": best_dice,
            "no_improve": no_improve,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="U-Net Background Removal Training")
    parser.add_argument("--config", required=True, help="Path to yaml config file")
    parser.add_argument(
        "--resume", action="store_true", help="Continue from <output_dir>/last.pth"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    last_path = os.path.join(out_dir, "last.pth")
    best_path = os.path.join(out_dir, "best.pth")
    hist_path = os.path.join(out_dir, "history.csv")
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    train_loader, val_loader, test_loader = get_dataloaders(
        root=cfg["data_root"],
        image_size=cfg["image_size"],
        batch_size=cfg["batch_size"],
        val_fraction=cfg["val_fraction"],
        seed=cfg["seed"],
        num_workers=cfg["num_workers"],
    )

    model = ResNetUnet(pretrained=cfg["pretrained"]).to(device)
    criterion = build_loss(cfg["loss"])
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_dice, no_improve = 1, 0.0, 0
    if args.resume:
        ckpt = torch.load(last_path, map_location=device)
        unwrap(model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_dice = ckpt["best_dice"]
        no_improve = ckpt["no_improve"]
        print(f"Resumed from epoch {ckpt['epoch']} (best dice {best_dice:.4f})")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    fields = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_dice",
        "val_iou",
        "lr",
        "encoder_lr",
        "frozen",
        "time_s",
    ]
    resuming = args.resume and os.path.exists(hist_path)
    hist_file = open(hist_path, "a" if resuming else "w", newline="")
    writer = csv.DictWriter(hist_file, fieldnames=fields)
    if not resuming:
        writer.writeheader()

    freeze_epochs = cfg["freeze_epochs"]
    max_grad_norm = cfg["max_grad_norm"]
    patience = cfg["patience"]

    print(
        f"device={device} amp={use_amp} gpus={torch.cuda.device_count()} "
        f"train_samples={len(train_loader.dataset)} val_samples={len(val_loader.dataset)} "
        f"freeze_epochs={freeze_epochs} max_grad_norm={max_grad_norm}"
    )

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t0 = time.time()

        # Manage backbone freezing
        is_frozen = epoch <= freeze_epochs
        set_encoder_freeze(model, freeze=is_frozen)
        if epoch == freeze_epochs + 1 and freeze_epochs > 0:
            print(f"[Epoch {epoch}] Unfreezing ResNet encoder backbone for joint fine-tuning.")

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp,
            max_grad_norm=max_grad_norm,
        )

        val_loss, val_dice, val_iou = evaluate(
            model, val_loader, criterion, device, use_amp
        )

        # Current learning rates
        current_enc_lr = optimizer.param_groups[0]["lr"]
        current_dec_lr = optimizer.param_groups[1]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_dice": round(val_dice, 5),
            "val_iou": round(val_iou, 5),
            "lr": round(current_dec_lr, 7),
            "encoder_lr": round(current_enc_lr, 7) if not is_frozen else 0.0,
            "frozen": int(is_frozen),
            "time_s": round(time.time() - t0, 1),
        }
        writer.writerow(row)
        hist_file.flush()

        # Step the learning rate scheduler
        if scheduler is not None:
            scheduler.step()

        status_str = (
            f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} | "
            f"dice {val_dice:.4f} | iou {val_iou:.4f} | lr {current_dec_lr:.2e} "
            f"{' [ENCODER FROZEN]' if is_frozen else ''}| {row['time_s']}s"
        )
        print(status_str)

        if val_dice > best_dice:
            best_dice, no_improve = val_dice, 0
            torch.save(unwrap(model).state_dict(), best_path)
        else:
            no_improve += 1

        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_dice,
            no_improve,
        )

        if no_improve >= patience:
            print(f"Early stopping: no val dice improvement for {patience} epochs")
            break

    hist_file.close()

    # Final score on the test split using best checkpoint
    unwrap(model).load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_dice, test_iou = evaluate(
        model, test_loader, criterion, device, use_amp
    )
    print(f"TEST | loss {test_loss:.4f} | dice {test_dice:.4f} | iou {test_iou:.4f}")
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(
            {
                "loss": test_loss,
                "dice": test_dice,
                "iou": test_iou,
                "best_val_dice": best_dice,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()