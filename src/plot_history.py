"""
Visualization utility for plotting background removal training progress.
Reads history.csv and generates multi-panel metric curves for:
1. Training and Validation Loss
2. Validation Dice and IoU Scores
3. Learning Rate Schedule (Decoder and Encoder)
4. Per-epoch training time
"""

import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_training_history(csv_path: str, output_path: str = None, show: bool = False):
    """
    Parses history.csv and generates a comprehensive 4-panel training plot.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Training history file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"Warning: {csv_path} is empty.")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = df["epoch"]

    # 1. Loss Curves
    ax_loss = axes[0, 0]
    ax_loss.plot(epochs, df["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax_loss.plot(epochs, df["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    ax_loss.set_title("Training & Validation Loss (BCE + Dice)", fontsize=13, fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, linestyle=":", alpha=0.6)
    ax_loss.legend(loc="upper right")

    # 2. Metric Curves (Dice & IoU)
    ax_metric = axes[0, 1]
    ax_metric.plot(epochs, df["val_dice"], label="Val Dice", color="#2ca02c", lw=2)
    ax_metric.plot(epochs, df["val_iou"], label="Val IoU", color="#9467bd", lw=2, linestyle="--")
    
    # Highlight best Dice score
    best_dice = df["val_dice"].max()
    best_epoch = df.loc[df["val_dice"].idxmax(), "epoch"]
    ax_metric.scatter([best_epoch], [best_dice], color="red", s=80, zorder=5)
    ax_metric.annotate(
        f"Best Dice: {best_dice:.4f} (Ep {int(best_epoch)})",
        (best_epoch, best_dice),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontweight="bold",
        color="red",
    )
    ax_metric.set_title("Segmentation Accuracy (Dice & IoU)", fontsize=13, fontweight="bold")
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel("Score (0.0 - 1.0)")
    ax_metric.set_ylim([0.0, 1.05])
    ax_metric.grid(True, linestyle=":", alpha=0.6)
    ax_metric.legend(loc="lower right")

    # 3. Learning Rate Curves
    ax_lr = axes[1, 0]
    if "lr" in df.columns:
        ax_lr.plot(epochs, df["lr"], label="Decoder LR", color="#d62728", lw=2)
    if "encoder_lr" in df.columns:
        ax_lr.plot(epochs, df["encoder_lr"], label="Encoder LR", color="#8c564b", lw=2, linestyle=":")
    ax_lr.set_title("Learning Rate Schedule", fontsize=13, fontweight="bold")
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("Learning Rate")
    ax_lr.set_yscale("log")
    ax_lr.grid(True, linestyle=":", alpha=0.6)
    ax_lr.legend(loc="upper right")

    # 4. Training Time per Epoch
    ax_time = axes[1, 1]
    if "time_s" in df.columns:
        ax_time.bar(epochs, df["time_s"], color="#17becf", alpha=0.7, edgecolor="#0e6670")
        avg_time = df["time_s"].mean()
        ax_time.axhline(avg_time, color="navy", linestyle="--", label=f"Avg: {avg_time:.1f}s")
        ax_time.set_title("Epoch Duration (Seconds)", fontsize=13, fontweight="bold")
        ax_time.set_xlabel("Epoch")
        ax_time.set_ylabel("Seconds")
        ax_time.grid(True, linestyle=":", alpha=0.6)
        ax_time.legend(loc="upper right")

    plt.suptitle("Background Removal (ResNet-34 U-Net) Training Report", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved training curves plot to: {output_path}")

    if show:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot training history metrics")
    parser.add_argument("--csv", default="outputs/history.csv", help="Path to history.csv")
    parser.add_argument("--output", default="outputs/training_curves.png", help="Path to output plot PNG")
    parser.add_argument("--show", action="store_true", help="Display interactive plot window")
    args = parser.parse_args()

    plot_training_history(args.csv, args.output, args.show)


if __name__ == "__main__":
    main()
