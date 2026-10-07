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


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
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
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap(model):
    """Return the underlying model if it is wrapped in DataParallel."""
    return model.module if isinstance(model, nn.DataParallel) else model


def build_loss(name):
    losses = {"bce_dice": BCEDiceLoss}
    if name not in losses:
        raise ValueError(f"Unknown loss '{name}'. Options: {list(losses)}")
    return losses[name]()


def build_optimizer(model, cfg):
    # pretrained encoder gets a smaller learning rate than the new decoder + head
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    return torch.optim.Adam(
        [
            {"params": enc, "lr": cfg["encoder_lr"]},
            {"params": rest, "lr": cfg["lr"]},
        ]
    )

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total, n = 0.0, 0
    for images, masks, valid in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
        # compute the loss in float32: fp16 sums over 256*256 pixels can overflow
        loss = criterion(logits.float(), masks, valid)

        scaler.scale(loss).backward()
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


def save_checkpoint(path, model, optimizer, scaler, epoch, best_dice, no_improve):
    torch.save(
        {
            "model": unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_dice": best_dice,
            "no_improve": no_improve,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true",
                        help="continue from <output_dir>/last.pth")
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
        yaml.safe_dump(cfg, f)  # keep the exact settings next to the results

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
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_dice, no_improve = 1, 0.0, 0
    if args.resume:
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_dice = ckpt["best_dice"]
        no_improve = ckpt["no_improve"]
        print(f"Resumed from epoch {ckpt['epoch']} (best dice {best_dice:.4f})")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # wrap AFTER building the optimizer

    fields = ["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "time_s"]
    resuming = args.resume and os.path.exists(hist_path)
    hist_file = open(hist_path, "a" if resuming else "w", newline="")
    writer = csv.DictWriter(hist_file, fieldnames=fields)
    if not resuming:
        writer.writeheader()

    patience = cfg.get("patience", 10)  # optional yaml key
    print(f"device={device} amp={use_amp} gpus={torch.cuda.device_count()} "
          f"train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     scaler, device, use_amp)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion,
                                               device, use_amp)

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_dice": round(val_dice, 5),
            "val_iou": round(val_iou, 5),
            "time_s": round(time.time() - t0, 1),
        }
        writer.writerow(row)
        hist_file.flush()
        print(f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} | "
              f"dice {val_dice:.4f} | iou {val_iou:.4f} | {row['time_s']}s")

        if val_dice > best_dice:
            best_dice, no_improve = val_dice, 0
            torch.save(unwrap(model).state_dict(), best_path)
        else:
            no_improve += 1

        save_checkpoint(last_path, model, optimizer, scaler, epoch,
                        best_dice, no_improve)

        if no_improve >= patience:
            print(f"Early stopping: no val dice improvement for {patience} epochs")
            break

    hist_file.close()

    # final score on the untouched test split, using the best checkpoint
    unwrap(model).load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_dice, test_iou = evaluate(model, test_loader, criterion,
                                              device, use_amp)
    print(f"TEST | loss {test_loss:.4f} | dice {test_dice:.4f} | iou {test_iou:.4f}")
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump({"loss": test_loss, "dice": test_dice, "iou": test_iou,
                   "best_val_dice": best_dice}, f, indent=2)


if __name__ == "__main__":
    main()