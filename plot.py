"""
Simple utility to plot training loss and validation Dice / IoU scores from history.csv.
"""

import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_history(csv_path: str = "outputs/history.csv", output_png: str = "outputs/curves.png"):
    if not os.path.isfile(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("CSV is empty.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = df["epoch"]

    # Loss Curves
    ax1.plot(epochs, df["train_loss"], label="Train Loss", lw=2, color="#1f77b4")
    ax1.plot(epochs, df["val_loss"], label="Val Loss", lw=2, color="#ff7f0e", linestyle="--")
    ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Metric Curves
    ax2.plot(epochs, df["val_dice"], label="Val Dice", lw=2, color="#2ca02c")
    ax2.plot(epochs, df["val_iou"], label="Val IoU", lw=2, color="#9467bd", linestyle="--")
    ax2.set_title("Validation Accuracy (Dice & IoU)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score (0.0 - 1.0)")
    ax2.set_ylim([0.0, 1.05])
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=180)
    print(f"Saved plot to: {output_png}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="outputs/history.csv", help="Path to history.csv")
    parser.add_argument("--out", default="outputs/curves.png", help="Path to output plot PNG")
    args = parser.parse_args()
    plot_history(args.csv, args.out)
