"""
Inference script for Deep Learning Background Removal.

Supports:
- Single image or entire directory processing
- Aspect ratio preserving letterbox preprocessing
- Transparent RGBA PNG output (default)
- Solid color background replacement (--bg-color R G B)
- Custom background image replacement (--bg-image path/to/bg.jpg)
- Standalone mask saving (--save-mask)
- Soft alpha blending vs hard thresholding
"""

import argparse
import glob
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from src.models.unet import ResNetUnet

# ImageNet statistics matching the pretrained ResNet-34 encoder
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")


def preprocess(
    image: Image.Image, target_size: int = 256
) -> Tuple[torch.Tensor, Tuple[int, int, int, int, int, int]]:
    """
    Resizes image with aspect ratio preservation (letterbox padding) to target_size x target_size.
    Returns:
        tensor: Normalized PyTorch tensor of shape [1, 3, target_size, target_size]
        pad_info: Tuple of (pad_left, pad_top, new_w, new_h, orig_w, orig_h)
    """
    orig_w, orig_h = image.size
    scale = target_size / max(orig_w, orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    resized = image.resize((new_w, new_h), Image.BILINEAR)

    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    pad_right = target_size - new_w - pad_left
    pad_bottom = target_size - new_h - pad_top

    padded = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    padded.paste(resized, (pad_left, pad_top))

    tensor = TF.to_tensor(padded)
    tensor = TF.normalize(tensor, MEAN, STD).unsqueeze(0)

    pad_info = (pad_left, pad_top, new_w, new_h, orig_w, orig_h)
    return tensor, pad_info


def postprocess_mask(
    pred_probs: np.ndarray, pad_info: Tuple[int, int, int, int, int, int]
) -> np.ndarray:
    """
    Removes letterbox padding and restores predicted mask to original image dimensions.
    Returns:
        alpha: 2D numpy array [H, W] with values in range [0.0, 1.0]
    """
    pad_left, pad_top, new_w, new_h, orig_w, orig_h = pad_info

    # Crop out the letterbox padding
    valid_mask = pred_probs[pad_top : pad_top + new_h, pad_left : pad_left + new_w]

    # Resize back to original resolution with bilinear interpolation for smooth edges
    mask_pil = Image.fromarray((np.clip(valid_mask, 0.0, 1.0) * 255.0).astype(np.uint8))
    mask_pil = mask_pil.resize((orig_w, orig_h), Image.BILINEAR)

    return np.array(mask_pil, dtype=np.float32) / 255.0


def load_model(weights_path: str, device: torch.device) -> ResNetUnet:
    """
    Loads ResNetUnet model and checkpoint weights, handling both raw state dicts
    and checkpoint dicts with DataParallel 'module.' prefixes.
    """
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Checkpoint weights not found: {weights_path}")

    model = ResNetUnet(pretrained=False)
    checkpoint = torch.load(weights_path, map_location=device)

    # Extract state dict if saved as a training dictionary
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Strip DataParallel 'module.' prefix if present
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict)
    model.to(device)
    model.eval()
    return model


def process_image(
    model: ResNetUnet,
    image_path: str,
    output_dir: str,
    device: torch.device,
    image_size: int = 256,
    threshold: float = 0.5,
    soft_matte: bool = True,
    bg_color: Optional[Tuple[int, int, int]] = None,
    bg_image_path: Optional[str] = None,
    save_mask: bool = False,
):
    """Processes a single image, predicts background removal mask, and saves results."""
    raw_img = Image.open(image_path).convert("RGB")
    tensor, pad_info = preprocess(raw_img, target_size=image_size)
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Reconstruct full-resolution alpha matte
    alpha = postprocess_mask(probs, pad_info)

    if not soft_matte:
        alpha = (alpha >= threshold).astype(np.float32)

    orig_np = np.array(raw_img, dtype=np.float32)
    alpha_3d = alpha[:, :, None]

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save mask if requested
    if save_mask:
        mask_out_path = os.path.join(output_dir, f"{base_name}_mask.png")
        mask_img = Image.fromarray((alpha * 255.0).astype(np.uint8))
        mask_img.save(mask_out_path)
        print(f"Saved mask: {mask_out_path}")

    # 2. Composite with custom background image
    if bg_image_path is not None:
        bg_raw = Image.open(bg_image_path).convert("RGB").resize(raw_img.size, Image.BILINEAR)
        bg_np = np.array(bg_raw, dtype=np.float32)
        composite = (orig_np * alpha_3d + bg_np * (1.0 - alpha_3d)).astype(np.uint8)
        out_path = os.path.join(output_dir, f"{base_name}_composite.jpg")
        Image.fromarray(composite).save(out_path, quality=95)
        print(f"Saved composite: {out_path}")

    # 3. Composite with solid RGB color
    elif bg_color is not None:
        bg_np = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3)
        composite = (orig_np * alpha_3d + bg_np * (1.0 - alpha_3d)).astype(np.uint8)
        out_path = os.path.join(output_dir, f"{base_name}_colored_bg.jpg")
        Image.fromarray(composite).save(out_path, quality=95)
        print(f"Saved colored background: {out_path}")

    # 4. Standard transparent RGBA PNG
    else:
        alpha_channel = (alpha * 255.0).astype(np.uint8)
        rgb_channels = np.clip(orig_np, 0, 255).astype(np.uint8)
        rgba = np.dstack([rgb_channels, alpha_channel])
        out_path = os.path.join(output_dir, f"{base_name}_nobg.png")
        Image.fromarray(rgba, mode="RGBA").save(out_path)
        print(f"Saved transparent PNG: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Background Removal Inference Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights (.pth checkpoint)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to single image file or directory containing images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/predictions",
        help="Directory to save output images",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Model input resolution (square)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binarization threshold when not using soft matting",
    )
    parser.add_argument(
        "--hard",
        action="store_true",
        help="Use hard binary mask instead of continuous soft alpha matte",
    )
    parser.add_argument(
        "--bg-color",
        nargs=3,
        type=int,
        default=None,
        metavar=("R", "G", "B"),
        help="Replace background with solid RGB color (e.g., --bg-color 255 255 255)",
    )
    parser.add_argument(
        "--bg-image",
        type=str,
        default=None,
        help="Path to a background image to composite behind the foreground",
    )
    parser.add_argument(
        "--save-mask",
        action="store_true",
        help="Also export the grayscale alpha mask as a separate image",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on ('cuda' or 'cpu'). Defaults to auto-detection.",
    )

    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading weights from: {args.weights} (using device: {device})")
    model = load_model(args.weights, device)

    # Collect input files
    if os.path.isdir(args.input):
        files = []
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(glob.glob(os.path.join(args.input, f"*{ext}")))
            files.extend(glob.glob(os.path.join(args.input, f"*{ext.upper()}")))
        files = sorted(list(set(files)))
    elif os.path.isfile(args.input):
        files = [args.input]
    else:
        raise FileNotFoundError(f"Input path not found: {args.input}")

    if not files:
        print(f"No supported images found in: {args.input}")
        return

    print(f"Found {len(files)} image(s) to process. Outputs will be saved to: {args.output}")

    bg_color_tuple = tuple(args.bg_color) if args.bg_color is not None else None

    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Processing {os.path.basename(file_path)}...")
        process_image(
            model=model,
            image_path=file_path,
            output_dir=args.output,
            device=device,
            image_size=args.image_size,
            threshold=args.threshold,
            soft_matte=not args.hard,
            bg_color=bg_color_tuple,
            bg_image_path=args.bg_image,
            save_mask=args.save_mask,
        )

    print("Inference completed successfully.")


if __name__ == "__main__":
    main()
