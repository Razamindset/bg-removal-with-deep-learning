import glob
import hashlib
import os
import random
from typing import List, Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
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

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        image_size: int = 256,
        augment: bool = False,
        rotation_degrees: float = 0.0,
        resize: bool = True,
    ):
        self.pairs = pairs
        self.image_size = image_size
        self.augment = augment
        self.rotation_degrees = rotation_degrees
        self.resize = resize
        self.jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # Always convert image to RGB and mask to single-channel Grayscale (L)
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.resize:
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

            # 2. Optional random rotation. This is CPU-heavy, so keep it at 0 for throughput.
            if self.rotation_degrees > 0:
                angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
                img_tensor = TF.rotate(img_tensor, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask_tensor = TF.rotate(mask_tensor, angle, interpolation=TF.InterpolationMode.BILINEAR)

            # 3. Photometric jitter on image only
            img_tensor = self.jitter(img_tensor)

        # Normalize image using ImageNet statistics
        img_tensor = TF.normalize(img_tensor, MEAN, STD)

        return img_tensor, mask_tensor


def _default_cache_dir(cfg) -> str:
    if cfg.cache_dir:
        return cfg.cache_dir
    return os.path.join(cfg.output_dir, f"cache_{cfg.image_size}")


def _prepare_resized_cache(
    pairs: List[Tuple[str, str]],
    image_size: int,
    cache_dir: str,
) -> List[Tuple[str, str]]:
    """
    Resizes source image/mask pairs once and reuses them across epochs.
    This avoids repeated large-image decode + resize work inside every DataLoader worker.
    """
    image_dir = os.path.join(cache_dir, "images")
    mask_dir = os.path.join(cache_dir, "masks")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    cached_pairs = []
    size = (image_size, image_size)
    for idx, (img_path, mask_path) in enumerate(tqdm(pairs, desc="Preparing resized cache")):
        pair_key = f"{img_path}|{mask_path}".encode("utf-8")
        path_hash = hashlib.sha1(pair_key).hexdigest()[:12]
        base = f"{idx:06d}_{path_hash}_{os.path.splitext(os.path.basename(img_path))[0]}"
        cached_img = os.path.join(image_dir, f"{base}.jpg")
        cached_mask = os.path.join(mask_dir, f"{base}.png")

        if not (os.path.isfile(cached_img) and os.path.isfile(cached_mask)):
            with Image.open(img_path) as img:
                img = img.convert("RGB").resize(size, Image.BILINEAR)
                img.save(cached_img, quality=95, optimize=False)
            with Image.open(mask_path) as mask:
                mask = mask.convert("L").resize(size, Image.BILINEAR)
                mask.save(cached_mask)

        cached_pairs.append((cached_img, cached_mask))

    return cached_pairs


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

    if cfg.cache_resized:
        cache_dir = _default_cache_dir(cfg)
        print(f"Preparing resized dataset cache at: {cache_dir}")
        pairs = _prepare_resized_cache(pairs, cfg.image_size, cache_dir)

    # Deterministic split
    random.seed(cfg.seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * cfg.val_split))
    val_pairs = shuffled[:n_val]
    train_pairs = shuffled[n_val:]
    resize_in_dataset = not cfg.cache_resized

    train_ds = HumanMattingDataset(
        train_pairs,
        image_size=cfg.image_size,
        augment=True,
        rotation_degrees=cfg.rotation_degrees,
        resize=resize_in_dataset,
    )
    val_ds = HumanMattingDataset(
        val_pairs,
        image_size=cfg.image_size,
        augment=False,
        resize=resize_in_dataset,
    )

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
