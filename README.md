# Deep Learning Background Removal with ResNet-34 U-Net

An end-to-end deep learning background removal / salient object matting system based on a U-Net architecture with a pretrained ResNet-34 backbone and custom decoder layers.

---

## Architecture Overview

- **Encoder**: Pretrained ImageNet ResNet-34 feature extractor.
- **Decoder**: Multi-stage transposed convolution / double convolution blocks with skip connections from encoder levels.
- **Loss Function**: Masked Binary Cross Entropy (BCE) + Soft Dice Loss (`BCEDiceLoss`).
- **Optimization**:
  - Backbone warmup / freezing (`freeze_epochs: 3`) to stabilize decoder initialization.
  - Cosine Annealing learning rate schedule (`CosineAnnealingLR`).
  - Gradient norm clipping to prevent gradient explosions with AMP mixed precision.
  - Early stopping monitoring validation Dice score.

---

## Project Structure

```
├── configs/
│   ├── pets.yaml                     # Full production training configuration
│   └── smoke_test.yaml               # 2-epoch dry run sanity check configuration
├── notebooks/
│   └── kaggle_training_pipeline.ipynb # Complete step-by-step Kaggle notebook
├── src/
│   ├── dataset/
│   │   └── pets.py                   # Oxford-IIIT Pet dataset loader with auto path resolution
│   ├── models/
│   │   ├── double_conv.py            # Double Conv block (Conv-BN-ReLU x2)
│   │   ├── encoder.py                # ResNet-34 feature extractor
│   │   ├── decoder.py                # Upsampling + skip concatenation block
│   │   └── unet.py                   # Complete ResNetUnet architecture
│   ├── losses.py                     # Masked BCE + Dice loss
│   ├── metrics.py                    # Dice and IoU evaluation metrics
│   ├── plot_history.py               # Visualizer for loss, Dice, IoU & LR curves
│   ├── predict.py                    # Production inference pipeline (RGBA, white bg, etc.)
│   └── train.py                      # Multi-epoch training engine
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation
```

---

## Installation

```bash
git clone https://github.com/Razamindset/bg-removal-with-deep-learning.git
cd bg-removal-with-deep-learning
pip install -r requirements.txt
```

---

## Training on Kaggle

The recommended environment for training is **Kaggle** (2x T4 GPUs):

1. Upload or clone the repository into Kaggle (`/kaggle/working`).
2. Attach **The Oxford-IIIT Pet Dataset** as an input, or enable **Internet: ON** in notebook settings.
3. Open and run [`notebooks/kaggle_training_pipeline.ipynb`](notebooks/kaggle_training_pipeline.ipynb):
   - **Step 1: Sanity Dry Run**: Runs 2 epochs with `configs/smoke_test.yaml`.
   - **Step 2: Full Production Training**: Runs 100 epochs with `configs/pets.yaml`.
   - **Step 3: Training Graphs**: Generates and plots multi-panel metric curves.
   - **Step 4: Inference**: Generates transparent cutouts and studio white backgrounds.
   - **Step 5: Bundle Artifacts**: Compresses checkpoints and predictions into a downloadable zip file.

---

## Local Training

### Quick 2-Epoch Sanity Test:
```bash
python -m src.train --config configs/smoke_test.yaml
```

### Full Production Training:
```bash
python -m src.train --config configs/pets.yaml
```

### Resume Interrupted Training:
```bash
python -m src.train --config configs/pets.yaml --resume
```

---

## Plotting Metrics & Learning Curves

Generate training curves from `history.csv`:
```bash
python -m src.plot_history --csv outputs/history.csv --output outputs/training_curves.png
```

---

## Inference / Background Removal

The inference pipeline in [`src/predict.py`](src/predict.py) preserves aspect ratios with letterboxing, upsamples smooth alpha boundaries, and produces cutouts.

### 1. Transparent RGBA PNG (Default)
```bash
python -m src.predict --weights outputs/best.pth --input path/to/image.jpg --output outputs/predictions
```

### 2. Process an Entire Folder & Export Masks
```bash
python -m src.predict --weights outputs/best.pth --input path/to/folder --output outputs/predictions --save-mask
```

### 3. Replace Background with Studio White
```bash
python -m src.predict --weights outputs/best.pth --input path/to/image.jpg --bg-color 255 255 255
```

### 4. Replace Background with Custom Image
```bash
python -m src.predict --weights outputs/best.pth --input path/to/image.jpg --bg-image path/to/background.jpg
```
