import os
import pathlib
import random
from typing import Tuple

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

# ImageNet stats expected by the pretrained ResNet34
# (same values as models.ResNet34_Weights.DEFAULT.transforms().mean / .std)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def resolve_dataset_root(root: str) -> Tuple[str, bool]:
    """
    Checks if OxfordIIITPet already exists locally or in common Kaggle/Colab directories.
    Returns:
        (resolved_root, should_download)
    If images already exist, should_download is set to False to avoid read-only
    filesystem PermissionErrors (common on Kaggle /kaggle/input/) or network timeouts.
    """
    candidates = [
        pathlib.Path(root),
        pathlib.Path("./data"),
        pathlib.Path("/kaggle/input/the-oxfordiiit-pet-dataset"),
        pathlib.Path("/kaggle/input/oxford-iiit-pet"),
        pathlib.Path("/kaggle/working/data"),
    ]

    for p in candidates:
        if not p.exists():
            continue

        # Case 1: p / oxford-iiit-pet / images exists
        if (p / "oxford-iiit-pet" / "images").is_dir() and (p / "oxford-iiit-pet" / "annotations").is_dir():
            return str(p), False

        # Case 2: p itself is oxford-iiit-pet (contains images & annotations directly)
        if (p / "images").is_dir() and (p / "annotations").is_dir():
            if p.name == "oxford-iiit-pet":
                return str(p.parent), False
            # If folder name is different (e.g. 'the-oxfordiiit-pet-dataset'), link/point appropriately
            return str(p.parent), False

    # Not found locally: return requested root and attempt download
    return root, True


class PetDataset(Dataset):
    def __init__(self, root, split="trainval", image_size=256, augment=False, download=True):
        super().__init__()
        resolved_root, auto_download = resolve_dataset_root(root)
        final_download = download and auto_download

        try:
            self.dataset = datasets.OxfordIIITPet(
                root=resolved_root, split=split, target_types="segmentation", download=final_download
            )
        except Exception as e:
            if final_download:
                raise RuntimeError(
                    f"Failed to download/load OxfordIIITPet at '{resolved_root}': {e}\n"
                    "Tip for Kaggle: Make sure Internet is ON in the right sidebar (Settings -> Internet: ON), "
                    "or attach 'The Oxford-IIIT Pet Dataset' as an Input to your notebook."
                ) from e
            raise

        self.image_size = image_size
        self.augment = augment
        self.jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, mask = self.dataset[index]
        img = img.convert("RGB")

        size = (self.image_size, self.image_size)
        img = img.resize(size, Image.BILINEAR)
        mask = mask.resize(size, Image.NEAREST)

        img = TF.to_tensor(img)  # [3, H, W] in 0..1

        trimap = np.array(mask)                       # values 1, 2, 3
        target = (trimap == 1).astype(np.float32)     # pet
        valid  = (trimap != 3).astype(np.float32)     # 0 on the border, 1 elsewhere

        mask  = torch.from_numpy(target).unsqueeze(0)
        valid = torch.from_numpy(valid).unsqueeze(0)

        if self.augment:
            # geometric: same transform on image and mask
            if random.random() < 0.5:
                img, mask, valid = TF.hflip(img), TF.hflip(mask), TF.hflip(valid)
            angle = random.uniform(-15, 15)
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
            valid = TF.rotate(valid, angle, interpolation=TF.InterpolationMode.NEAREST)

            # photometric: image only, and while it is still in 0..1
            img = self.jitter(img)

        # normalize last, image only
        img = TF.normalize(img, MEAN, STD)
        return img, mask, valid


def get_dataloaders(root, image_size=256, batch_size=32, val_fraction=0.1,
                    seed=42, num_workers=2):
    # Resolve root once
    resolved_root, should_download = resolve_dataset_root(root)

    # First load train set (handles download if needed)
    train_full = PetDataset(resolved_root, "trainval", image_size, augment=True, download=should_download)
    # Subsequent loads reuse data with download=False
    val_full = PetDataset(resolved_root, "trainval", image_size, augment=False, download=False)
    test_set = PetDataset(resolved_root, "test", image_size, augment=False, download=False)

    # seeded split, so every run uses the same train/val images
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_full), generator=g).tolist()
    n_val = int(len(perm) * val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_loader = DataLoader(Subset(train_full, train_idx), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(Subset(val_full, val_idx), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders("./data", batch_size=4)
    images, masks, valids = next(iter(train_loader))
    print(images.shape, images.min().item(), images.max().item())
    print(masks.shape, masks.dtype, torch.unique(masks))
    print(valids.shape, valids.dtype, torch.unique(valids))
    print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))