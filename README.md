# Deep Learning Background Removal (ResNet-34 U-Net)

A streamlined, production-grade deep learning background removal system specialized for human portraits and objects. Built with a U-Net architecture featuring a pretrained ResNet-34 backbone and an artifact-free bilinear decoder.

---

## Key Highlights

- **Pretrained Backbone**: ResNet-34 ImageNet feature extractor with warmup freezing.
- **Smooth Bilinear Decoder**: Replaces transposed convolutions with bilinear upsampling + conv to eliminate checkerboard edge artifacts on hair and body boundaries.
- **Loss & Metrics**: Soft Dice Loss + Binary Cross Entropy (`BCEDiceLoss`), evaluated with Dice score and IoU.
- **Multi-GPU Ready**: Automatically detects multiple GPUs (e.g. 2x T4 on Kaggle) and enables `torch.nn.DataParallel`.
- **Flat & Clean Codebase**: No complex YAML parsers, no deep folder nesting. Every file is self-contained and easy to read.

---

## Clean Project Structure

```
├── config.py         # All hyperparameters and paths in one clean Python class
├── dataset.py        # Paired image + mask/matte dataset loader with augmentations
├── model.py          # Unified U-Net architecture (ResNet-34 + Bilinear Decoder)
├── loss.py           # Combined BCEDiceLoss and Dice/IoU evaluation metrics
├── train.py          # Training loop with Multi-GPU, CosineAnnealingLR, and early stopping
├── predict.py        # Inference pipeline (transparent PNGs, solid backgrounds)
├── plot.py           # Script to plot Loss, Dice, and IoU curves from history.csv
├── notebook.ipynb    # End-to-end Kaggle training & inference walkthrough
├── requirements.txt  # Python package dependencies
└── README.md         # Documentation
```

---

## Installation

```bash
git clone https://github.com/Razamindset/bg-removal-with-deep-learning.git
cd bg-removal-with-deep-learning
pip install -r requirements.txt
```

---

## Quick Start

### 1. Training on Kaggle (Recommended)
Open and run [`notebook.ipynb`](notebook.ipynb) on Kaggle:
- Supports 2x T4 GPUs out of the box with `nn.DataParallel`.
- Set dataset path in `config.py` (e.g., `/kaggle/input/p3m-10k`).
- Automatically handles asset caching, training, curve plotting, sample visual cutouts, and artifact compression.

### 2. Training Locally
```bash
python train.py
```
- Automatically uses GPU if available, or CPU.
- Checkpoints are saved to `outputs/best_model.pth` and `outputs/last_model.pth`.
- History is logged to `outputs/history.csv`.

### 3. Plot Training Curves
```bash
python plot.py --csv outputs/history.csv --out outputs/curves.png
```

### 4. Background Removal Inference

#### Extract Transparent RGBA Cutout (Default):
```bash
python predict.py --weights outputs/best_model.pth --input path/to/image.jpg --output outputs/predictions
```

#### Process an Entire Folder:
```bash
python predict.py --weights outputs/best_model.pth --input path/to/folder --output outputs/predictions
```

#### Replace Background with Studio White:
```bash
python predict.py --weights outputs/best_model.pth --input path/to/image.jpg --bg-color 255 255 255
```

---

## Configuration

All settings can be customized in [`config.py`](config.py):
```python
from config import cfg

cfg.data_dir = "/kaggle/input/p3m-10k"   # Path to images and masks
cfg.image_size = 256                    # Input resolution
cfg.batch_size = 64                     # Batch size per step; keep >=16 samples per GPU
cfg.num_workers = 4                     # Increase if the GPU waits between batches
cfg.cache_resized = True                # Resize images/masks once, then train from cache
cfg.rotation_degrees = 0.0              # Set to 10.0 for stronger but slower augmentation
cfg.epochs = 40                         # Total training epochs
cfg.lr = 1e-3                           # Decoder learning rate
cfg.encoder_lr = 1e-4                   # Pretrained backbone learning rate
cfg.freeze_epochs = 2                   # Backbone warmup freeze epochs
```
