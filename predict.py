"""
Clean, straightforward inference script for Background Removal.
Takes an image or directory, removes background, and exports transparent PNGs.
"""

import argparse
import glob
import os
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from model import ResNetUNet

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def preprocess(image: Image.Image, size: int = 256) -> Tuple[torch.Tensor, Tuple]:
    """Resizes image with aspect ratio letterbox padding to (size, size)."""
    orig_w, orig_h = image.size
    scale = size / max(orig_w, orig_h)
    new_w, new_h = max(1, int(round(orig_w * scale))), max(1, int(round(orig_h * scale)))

    resized = image.resize((new_w, new_h), Image.BILINEAR)
    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2

    padded = Image.new("RGB", (size, size), (0, 0, 0))
    padded.paste(resized, (pad_left, pad_top))

    tensor = TF.to_tensor(padded)
    tensor = TF.normalize(tensor, MEAN, STD).unsqueeze(0)
    pad_info = (pad_left, pad_top, new_w, new_h, orig_w, orig_h)
    return tensor, pad_info


def postprocess_mask(prob_map: np.ndarray, pad_info: Tuple) -> np.ndarray:
    """Crops letterbox padding and restores mask back to original image resolution."""
    pad_left, pad_top, new_w, new_h, orig_w, orig_h = pad_info
    cropped = prob_map[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
    mask_pil = Image.fromarray((np.clip(cropped, 0.0, 1.0) * 255.0).astype(np.uint8))
    mask_pil = mask_pil.resize((orig_w, orig_h), Image.BILINEAR)
    return np.array(mask_pil, dtype=np.float32) / 255.0


def remove_background(model, image_path: str, output_path: str, device: torch.device, bg_color=None):
    raw_img = Image.open(image_path).convert("RGB")
    tensor, pad_info = preprocess(raw_img)
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    alpha = postprocess_mask(probs, pad_info)
    img_np = np.array(raw_img)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if bg_color is not None:
        # Solid background replacement (e.g., white [255, 255, 255])
        bg = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3)
        alpha_3d = alpha[:, :, None]
        composite = (img_np * alpha_3d + bg * (1.0 - alpha_3d)).astype(np.uint8)
        Image.fromarray(composite).save(output_path, quality=95)
    else:
        # Transparent RGBA PNG
        rgba = np.dstack([img_np, (alpha * 255.0).astype(np.uint8)])
        Image.fromarray(rgba, mode="RGBA").save(output_path)

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Human Background Removal Prediction")
    parser.add_argument("--weights", default="outputs/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--input", required=True, help="Path to input image or directory")
    parser.add_argument("--output", default="outputs/predictions", help="Directory to save predictions")
    parser.add_argument("--bg-color", nargs=3, type=int, default=None, help="Replace bg with RGB color e.g. 255 255 255")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetUNet(pretrained=False)
    state = torch.load(args.weights, map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # Collect images
    if os.path.isdir(args.input):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(args.input, e)))
    else:
        files = [args.input]

    if not files:
        print(f"No images found in {args.input}")
        return

    os.makedirs(args.output, exist_ok=True)
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        out_ext = ".png" if args.bg_color is None else ".jpg"
        out_p = os.path.join(args.output, f"{base}_nobg{out_ext}")
        remove_background(model, f, out_p, device, bg_color=args.bg_color)


if __name__ == "__main__":
    main()
