import glob
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ImageNet statistics for ResNet backbone
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def find_image_mask_pairs(data_dir: str) -> List[Tuple[str, str]]:
    """
    Finds (image_path, mask_path) pairs from common dataset directory layouts:
    Layout A: data_dir/images and data_dir/masks (or data_dir/mattes)
    Layout B: data_dir/train/original_image and data_dir/train/matte (P3M-10k style)
    Layout C: recursive search matching identical basenames
    """
    valid_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    all_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(valid_exts):
                all_files.append(os.path.join(root, f))

    # Separate candidates into masks/mattes and images
    masks = {}
    images = {}
    for p in all_files:
        p_lower = p.lower().replace("\\", "/")
        base_name = os.path.splitext(os.path.basename(p))[0]

        if "mask" in p_lower or "matte" in p_lower:
            masks[base_name] = p
        else:
            images[base_name] = p

    # Match by base file name
    pairs = []
    for base, img_path in images.items():
        if base in masks:
            pairs.append((img_path, masks[base]))

    # Fallback: if no keyword separation found, check adjacent 'images' and 'masks' folders
    if not pairs:
        img_folder = os.path.join(data_dir, "images")
        mask_folder = os.path.join(data_dir, "masks")
        if os.path.isdir(img_folder) and os.path.isdir(mask_folder):
            for img_p in glob.glob(os.path.join(img_folder, "*.*")):
                base = os.path.splitext(os.path.basename(img_p))[0]
                for ext in (".png", ".jpg", ".jpeg"):
                    candidate = os.path.join(mask_folder, f"{base}{ext}")
                    if os.path.isfile(candidate):
                        pairs.append((img_p, candidate))
                        break

    return sorted(pairs)


class HumanMattingDataset(Dataset):
    """
    Clean Dataset for Human & Portrait Background Removal.
    Loads paired (image, mask) and applies synchronized augmentations.
    """

    def __init__(self, pairs: List[Tuple[str, str]], image_size: int = 256, augment: bool = False):
        self.pairs = pairs
        self.image_size = image_size
        self.augment = augment
        self.jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # Always convert image to RGB and mask to single-channel Grayscale (L)
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize to model resolution
        size = (self.image_size, self.image_size)
        img = img.resize(size, Image.BILINEAR)
        mask = mask.resize(size, Image.BILINEAR)

        # Convert to tensors
        img_tensor = TF.to_tensor(img)                     # [3, H, W] in range 0..1
        mask_tensor = TF.to_tensor(mask)                   # [1, H, W] in range 0..1

        # Synchronized Data Augmentations
        if self.augment:
            # 1. Random horizontal flip
            if random.random() < 0.5:
                img_tensor = TF.hflip(img_tensor)
                mask_tensor = TF.hflip(mask_tensor)

            # 2. Random slight rotation (-10 to 10 degrees)
            angle = random.uniform(-10.0, 10.0)
            img_tensor = TF.rotate(img_tensor, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask_tensor = TF.rotate(mask_tensor, angle, interpolation=TF.InterpolationMode.BILINEAR)

            # 3. Photometric jitter on image only
            img_tensor = self.jitter(img_tensor)

        # Normalize image using ImageNet statistics
        img_tensor = TF.normalize(img_tensor, MEAN, STD)

        return img_tensor, mask_tensor


def get_dataloaders(cfg):
    """
    Builds clean training and validation DataLoaders.
    """
    pairs = find_image_mask_pairs(cfg.data_dir)
    if not pairs:
        raise FileNotFoundError(
            f"No matching (image, mask) pairs found in '{cfg.data_dir}'.\n"
            "Ensure the dataset directory contains images and corresponding masks/mattes."
        )

    # Deterministic split
    random.seed(cfg.seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * cfg.val_split))
    val_pairs = shuffled[:n_val]
    train_pairs = shuffled[n_val:]

    train_ds = HumanMattingDataset(train_pairs, image_size=cfg.image_size, augment=True)
    val_ds = HumanMattingDataset(val_pairs, image_size=cfg.image_size, augment=False)

    loader_kwargs = {
        "num_workers": cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": cfg.num_workers > 0,
    }
    if cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    print(f"Loaded {len(pairs)} total pairs: {len(train_ds)} train, {len(val_ds)} val")
    return train_loader, val_loader
