import torch
from dataclasses import dataclass

@dataclass
class Config:
    # --- Dataset & Paths ---
    # Common Kaggle path: /kaggle/input/p3m-10k or /kaggle/input/portrait-matting or ./data
    data_dir: str = "/kaggle/input/datasets/rahulbhalley/p3m-10k"
    image_size: int = 256
    val_split: float = 0.1
    batch_size: int = 64
    num_workers: int = 4
    seed: int = 42

    # --- Training Hyperparameters ---
    epochs: int = 40
    lr: float = 1e-3             # Decoder learning rate
    encoder_lr: float = 1e-4     # Pretrained ResNet encoder learning rate
    min_lr: float = 1e-6         # Minimum learning rate for cosine scheduler
    freeze_epochs: int = 2       # Initial epochs to keep encoder frozen (warmup)
    max_grad_norm: float = 1.0   # Gradient clipping threshold
    patience: int = 8            # Early stopping patience on validation Dice
    progress_interval: int = 20   # Update progress bars every N batches to avoid GPU sync overhead
    use_data_parallel: bool = True

    # --- Hardware & Outputs ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "outputs"

    def display(self):
        """Prints active configuration."""
        print("=" * 45)
        print("          TRAINING CONFIGURATION")
        print("=" * 45)
        for key, value in self.__dict__.items():
            print(f"  {key:<15}: {value}")
        print("=" * 45)


# Global default configuration instance
cfg = Config()
