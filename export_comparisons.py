"""
Automated Pipeline to Generate:
1. Individual 3-Panel Comparison Images (Original vs Predicted Mask vs Removed Background)
   for each input image in test-images/
2. LinkedIn-Ready Presentation Slide Deck (1920x1080)
3. Full Multi-Page Portfolio PDF including Training Curves & Case Studies.

Can re-run model inference on test images or use existing predictions.
"""

import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

from model import ResNetUNet

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_font(size: int, bold: bool = False):
    font_name = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(font_name, size)
    except IOError:
        try:
            return ImageFont.truetype("segoeui.ttf", size)
        except IOError:
            return ImageFont.load_default()


def make_checkerboard(width: int, height: int, sq_size: int = 20) -> Image.Image:
    """Generates a subtle high-contrast transparency grid."""
    c1 = (248, 250, 252)  # subtle light
    c2 = (226, 232, 240)  # subtle gray
    img = Image.new("RGB", (width, height), c1)
    draw = ImageDraw.Draw(img)
    for y in range(0, height, sq_size):
        for x in range(0, width, sq_size):
            if (x // sq_size + y // sq_size) % 2 == 1:
                draw.rectangle([x, y, min(x + sq_size, width), min(y + sq_size, height)], fill=c2)
    return img


def run_inference_on_image(model, raw_img: Image.Image, device: torch.device, size: int = 256) -> Image.Image:
    """Runs model forward pass and reconstructs full-res RGBA output."""
    orig_w, orig_h = raw_img.size
    scale = size / max(orig_w, orig_h)
    new_w, new_h = max(1, int(round(orig_w * scale))), max(1, int(round(orig_h * scale)))

    resized = raw_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2

    padded = Image.new("RGB", (size, size), (0, 0, 0))
    padded.paste(resized, (pad_left, pad_top))

    tensor = TF.to_tensor(padded)
    tensor = TF.normalize(tensor, MEAN, STD).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Crop letterbox padding and resize back to native resolution
    cropped = probs[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
    mask_pil = Image.fromarray((np.clip(cropped, 0.0, 1.0) * 255.0).astype(np.uint8))
    mask_pil = mask_pil.resize((orig_w, orig_h), Image.Resampling.BILINEAR)

    # Composite RGBA
    img_np = np.array(raw_img)
    alpha_np = np.array(mask_pil, dtype=np.uint8)
    rgba = np.dstack([img_np, alpha_np])
    return Image.fromarray(rgba, mode="RGBA")


def build_compact_triplet(orig_img: Image.Image, rgba_img: Image.Image, sample_name: str, target_h: int = 800) -> Image.Image:
    """
    Creates a dedicated standalone 3-panel image:
    [ Original Image ] vs [ Predicted Mask ] vs [ Removed Background ]
    """
    orig_w, orig_h = orig_img.size
    scale = target_h / orig_h
    tw, th = int(round(orig_w * scale)), target_h

    orig_res = orig_img.resize((tw, th), Image.Resampling.LANCZOS)
    rgba_res = rgba_img.resize((tw, th), Image.Resampling.LANCZOS)

    # Alpha mask
    alpha = rgba_res.split()[3]
    mask_res = Image.merge("RGB", (alpha, alpha, alpha))

    # Removed background over checkerboard
    sq = max(12, tw // 28)
    cb = make_checkerboard(tw, th, sq_size=sq)
    cb.paste(rgba_res, (0, 0), rgba_res)

    # Dimensions
    pad_x = 20
    pad_y = 20
    gap = 16
    header_h = 70
    footer_h = 36

    total_w = pad_x * 2 + tw * 3 + gap * 2
    total_h = pad_y * 2 + header_h + th + footer_h

    canvas = Image.new("RGB", (total_w, total_h), (15, 23, 42))  # slate-900
    draw = ImageDraw.Draw(canvas)

    f_title = load_font(24, bold=True)
    f_sub = load_font(14, bold=False)
    f_badge = load_font(16, bold=True)
    f_footer = load_font(13, bold=False)

    # Header text
    draw.text((pad_x, pad_y + 6), f"Sample: {sample_name} | Background Removal Pipeline", fill=(248, 250, 252), font=f_title)
    draw.text((pad_x, pad_y + 38), f"Native Resolution: {orig_w} \u00d7 {orig_h} px | Model: ResNet-34 U-Net (Bilinear)", fill=(148, 163, 184), font=f_sub)

    panels = [
        ("Original Image", orig_res, (59, 130, 246)),
        ("Predicted Mask", mask_res, (6, 182, 212)),
        ("Removed Background", cb, (16, 185, 129))
    ]

    start_y = pad_y + header_h
    for idx, (label, img_panel, badge_color) in enumerate(panels):
        px = pad_x + idx * (tw + gap)
        # Card container
        draw.rounded_rectangle([px - 4, start_y - 4, px + tw + 4, start_y + th + 4], radius=8, fill=(30, 41, 59), outline=(51, 65, 85), width=1)
        canvas.paste(img_panel, (px, start_y))

        # Badge
        badge_w = 190
        badge_h = 28
        bx0 = px + (tw - badge_w) // 2
        by0 = start_y - 34
        draw.rounded_rectangle([bx0, by0, bx0 + badge_w, by0 + badge_h], radius=14, fill=badge_color)
        bbox = draw.textbbox((0, 0), label, font=f_badge)
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((bx0 + (badge_w - bw) // 2, by0 + (badge_h - bh) // 2 - 1), label, fill=(255, 255, 255), font=f_badge)

    draw.text((pad_x, total_h - pad_y - 18), "Deep Learning Background Removal | Soft BCEDice Loss | Dice 98.4% | IoU 96.9%", fill=(148, 163, 184), font=f_footer)
    return canvas


def build_slide_cover() -> Image.Image:
    """Builds an executive presentation cover slide (1920x1080)."""
    canvas_w, canvas_h = 1920, 1080
    canvas = Image.new("RGB", (canvas_w, canvas_h), (15, 23, 42))
    draw = ImageDraw.Draw(canvas)

    f_tag = load_font(18, bold=True)
    f_h1 = load_font(52, bold=True)
    f_h2 = load_font(26, bold=False)
    f_stat_num = load_font(42, bold=True)
    f_stat_label = load_font(16, bold=True)
    f_stat_sub = load_font(14, bold=False)
    f_body = load_font(18, bold=False)
    f_footer = load_font(16, bold=False)

    # Accent glow top bar
    draw.rectangle([0, 0, canvas_w, 6], fill=(59, 130, 246))

    # Category Pill
    draw.rounded_rectangle([120, 100, 390, 140], radius=20, fill=(30, 58, 138), outline=(59, 130, 246), width=1)
    draw.text((140, 108), "DEEP LEARNING SHOWCASE", fill=(147, 197, 253), font=f_tag)

    # Main Titles
    draw.text((120, 170), "Production Background Removal", fill=(248, 250, 252), font=f_h1)
    draw.text((120, 240), "High-Fidelity Portrait & Object Matting with ResNet-34 U-Net", fill=(56, 189, 248), font=f_h1)
    draw.text((120, 325), "Artifact-Free Bilinear Decoder \u2022 Soft BCEDice Combined Loss \u2022 Multi-GPU PyTorch Pipeline", fill=(148, 163, 184), font=f_h2)

    # 4 Key Stat Cards
    stat_y = 420
    stat_w = 385
    stat_h = 150
    stat_gap = 46
    stats = [
        ("98.39%", "VAL DICE SCORE", "Harmonic mean of precision & recall", (16, 185, 129)),
        ("96.92%", "VAL IOU SCORE", "Intersection over Union (Jaccard)", (59, 130, 246)),
        ("0.0502", "FINAL LOSS", "Down from 0.4878 initial loss", (6, 182, 212)),
        ("40 Epochs", "COSINE LR SCHEDULE", "Smooth decay with backbone warmup", (168, 85, 247)),
    ]

    for idx, (num, label, desc, color) in enumerate(stats):
        cx = 120 + idx * (stat_w + stat_gap)
        draw.rounded_rectangle([cx, stat_y, cx + stat_w, stat_y + stat_h], radius=16, fill=(30, 41, 59), outline=(51, 65, 85), width=2)
        draw.rounded_rectangle([cx + 20, stat_y + 24, cx + 26, stat_y + stat_h - 24], radius=3, fill=color)
        draw.text((cx + 42, stat_y + 20), num, fill=(248, 250, 252), font=f_stat_num)
        draw.text((cx + 42, stat_y + 78), label, fill=color, font=f_stat_label)
        draw.text((cx + 42, stat_y + 104), desc, fill=(148, 163, 184), font=f_stat_sub)

    # 3 Engineering Architecture Cards
    arch_y = 620
    arch_w = 520
    arch_h = 270
    arch_gap = 60

    features = [
        ("Pretrained Feature Extractor", "ResNet-34 backbone pre-trained on ImageNet. Initial 2-epoch freeze warmup prevents disruptive gradient propagation into pretrained weights.", (59, 130, 246)),
        ("Smooth Bilinear Decoder", "Replaces standard transposed convolutions with bilinear interpolation + convolution, eliminating checkerboard boundary artifacts on hair and silhouette edges.", (6, 182, 212)),
        ("Compound Soft Loss & Matting", "Joint BCEDice loss directly penalizes boundary misalignments. Aspect-ratio preserving letterbox pipeline restores outputs up to 32+ Megapixels.", (16, 185, 129))
    ]

    for idx, (head, text, color) in enumerate(features):
        ax = 120 + idx * (arch_w + arch_gap)
        draw.rounded_rectangle([ax, arch_y, ax + arch_w, arch_y + arch_h], radius=16, fill=(24, 33, 47), outline=(51, 65, 85), width=1)
        draw.rounded_rectangle([ax + 24, arch_y + 24, ax + 140, arch_y + 56], radius=8, fill=(30, 41, 59))
        draw.text((ax + 36, arch_y + 30), f"PHASE 0{idx+1}", fill=color, font=f_stat_label)
        draw.text((ax + 24, arch_y + 75), head, fill=(248, 250, 252), font=load_font(22, bold=True))

        # Multi-line wrap
        words = text.split(" ")
        lines = []
        cur_line = []
        for w in words:
            cur_line.append(w)
            bbox = draw.textbbox((0, 0), " ".join(cur_line), font=f_body)
            if bbox[2] - bbox[0] > (arch_w - 48):
                cur_line.pop()
                lines.append(" ".join(cur_line))
                cur_line = [w]
        if cur_line:
            lines.append(" ".join(cur_line))

        for l_idx, line in enumerate(lines):
            draw.text((ax + 24, arch_y + 115 + l_idx * 28), line, fill=(148, 163, 184), font=f_body)

    # Footer
    draw.line([(120, 970), (1800, 970)], fill=(51, 65, 85), width=1)
    draw.text((120, 1000), "Deep Learning Engineering Portfolio | LinkedIn Document Post Presentation Deck", fill=(148, 163, 184), font=f_footer)
    draw.text((1560, 1000), "Swipe For Case Studies \u2192", fill=(56, 189, 248), font=load_font(18, bold=True))

    return canvas


def build_slide_curves(curves_path: str) -> Image.Image:
    """Builds the 1920x1080 slide displaying training curves and loss convergence."""
    canvas_w, canvas_h = 1920, 1080
    canvas = Image.new("RGB", (canvas_w, canvas_h), (15, 23, 42))
    draw = ImageDraw.Draw(canvas)

    f_title = load_font(38, bold=True)
    f_sub = load_font(20, bold=False)
    f_card_num = load_font(36, bold=True)
    f_card_label = load_font(16, bold=True)
    f_card_sub = load_font(14, bold=False)
    f_footer = load_font(18, bold=False)
    f_meta = load_font(16, bold=True)

    draw.text((80, 50), "Training Dynamics & Convergence Performance", fill=(248, 250, 252), font=f_title)
    draw.text((80, 105), "Quantitative Analysis: 40 Epochs with Cosine Annealing Learning Rate & Warmup Backbone Freezing", fill=(148, 163, 184), font=f_sub)
    draw.line([(80, 145), (1840, 145)], fill=(51, 65, 85), width=2)

    # 4 Metric Cards
    card_w = 405
    card_h = 110
    card_gap = 46
    start_x = 80
    start_y = 165

    metrics = [
        ("98.39%", "Peak Validation Dice", "Harmonic mean of precision & recall", (16, 185, 129)),
        ("96.92%", "Peak Validation IoU", "Jaccard index of segment overlap", (59, 130, 246)),
        ("0.0502", "Best Validation Loss", "Combined Soft BCEDice Loss", (6, 182, 212)),
        ("40 Epochs", "Cosine Annealing LR", "Decayed from 1e-3 down to 1e-6", (168, 85, 247))
    ]

    for idx, (val, title, desc, accent) in enumerate(metrics):
        cx = start_x + idx * (card_w + card_gap)
        cy = start_y
        draw.rounded_rectangle([cx, cy, cx + card_w, cy + card_h], radius=12, fill=(30, 41, 59), outline=(51, 65, 85), width=2)
        draw.rounded_rectangle([cx + 16, cy + 18, cx + 22, cy + card_h - 18], radius=3, fill=accent)
        draw.text((cx + 34, cy + 16), val, fill=(248, 250, 252), font=f_card_num)
        draw.text((cx + 34, cy + 58), title.upper(), fill=accent, font=f_card_label)
        draw.text((cx + 34, cy + 80), desc, fill=(148, 163, 184), font=f_card_sub)

    # Curves image
    if os.path.exists(curves_path):
        curves_img = Image.open(curves_path).convert("RGB")
        cw, ch = curves_img.size
        avail_w = 1760
        avail_h = 690
        scale = min(avail_w / cw, avail_h / ch)
        tw, th = int(round(cw * scale)), int(round(ch * scale))
        curves_res = curves_img.resize((tw, th), Image.Resampling.LANCZOS)

        px = 80 + (avail_w - tw) // 2
        py = 295 + (avail_h - th) // 2

        draw.rounded_rectangle([px - 6, py - 6, px + tw + 6, py + th + 6], radius=14, fill=(30, 41, 59), outline=(51, 65, 85), width=2)
        canvas.paste(curves_res, (px, py))

    # Footer
    draw.line([(80, 1010), (1840, 1010)], fill=(51, 65, 85), width=1)
    draw.text((80, 1025), "P3M-10k Dataset | ResNet-34 Feature Extractor | Bilinear Smooth Decoder", fill=(148, 163, 184), font=f_footer)
    draw.text((1600, 1025), "Visual Comparisons \u2192", fill=(59, 130, 246), font=f_meta)
    return canvas


def build_slide_case_study(orig_img: Image.Image, rgba_img: Image.Image, case_idx: int, case_title: str) -> Image.Image:
    """Builds a 1920x1080 slide showcasing [Original] vs [Mask] vs [Cutout]."""
    canvas_w, canvas_h = 1920, 1080
    canvas = Image.new("RGB", (canvas_w, canvas_h), (15, 23, 42))
    draw = ImageDraw.Draw(canvas)

    orig_w, orig_h = orig_img.size

    f_title = load_font(38, bold=True)
    f_sub = load_font(20, bold=False)
    f_badge = load_font(18, bold=True)
    f_footer = load_font(18, bold=False)
    f_meta = load_font(16, bold=True)

    draw.text((80, 50), f"Case Study 0{case_idx}: {case_title}", fill=(248, 250, 252), font=f_title)
    draw.text((80, 105), f"Native Input Resolution: {orig_w} \u00d7 {orig_h} px | Deep Learning Architecture: ResNet-34 U-Net (Bilinear Decoder)", fill=(148, 163, 184), font=f_sub)
    draw.line([(80, 145), (1840, 145)], fill=(51, 65, 85), width=2)

    # Panel sizing
    max_p_w = 540
    max_p_h = 750
    gap = 70

    scale = min(max_p_w / orig_w, max_p_h / orig_h)
    tw, th = int(round(orig_w * scale)), int(round(orig_h * scale))

    orig_res = orig_img.resize((tw, th), Image.Resampling.LANCZOS)
    pred_res = rgba_img.resize((tw, th), Image.Resampling.LANCZOS)
    alpha = pred_res.split()[3]
    mask_res = Image.merge("RGB", (alpha, alpha, alpha))

    cb = make_checkerboard(tw, th, sq_size=max(12, tw // 28))
    cb.paste(pred_res, (0, 0), pred_res)

    panels = [
        ("1. Original Image", orig_res, (59, 130, 246)),
        ("2. Predicted Alpha Mask", mask_res, (6, 182, 212)),
        ("3. Removed Background", cb, (16, 185, 129))
    ]

    actual_total_w = tw * 3 + gap * 2
    offset_x = (canvas_w - actual_total_w) // 2
    avail_h = 800
    offset_y = 170 + (avail_h - th) // 2

    for idx, (label, img_panel, badge_color) in enumerate(panels):
        px = offset_x + idx * (tw + gap)
        py = offset_y

        card_p = 10
        draw.rounded_rectangle([px - card_p, py - card_p, px + tw + card_p, py + th + card_p], radius=14, fill=(30, 41, 59), outline=(51, 65, 85), width=2)
        canvas.paste(img_panel, (px, py))

        badge_w = 240
        badge_h = 32
        bx0 = px + (tw - badge_w) // 2
        by0 = py - 46
        draw.rounded_rectangle([bx0, by0, bx0 + badge_w, by0 + badge_h], radius=16, fill=badge_color)
        bbox = draw.textbbox((0, 0), label, font=f_badge)
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((bx0 + (badge_w - bw) // 2, by0 + (badge_h - bh) // 2 - 1), label, fill=(255, 255, 255), font=f_badge)

    draw.line([(80, 1010), (1840, 1010)], fill=(51, 65, 85), width=1)
    draw.text((80, 1025), "ResNet-34 U-Net | Soft BCEDice Loss | Dice Score: 98.4% | IoU: 96.9%", fill=(148, 163, 184), font=f_footer)
    draw.text((1600, 1025), f"Slide {case_idx + 2} of 10", fill=(59, 130, 246), font=f_meta)

    return canvas


def build_slide_summary() -> Image.Image:
    """Builds the final conclusion / production readiness slide (1920x1080)."""
    canvas_w, canvas_h = 1920, 1080
    canvas = Image.new("RGB", (canvas_w, canvas_h), (15, 23, 42))
    draw = ImageDraw.Draw(canvas)

    f_title = load_font(46, bold=True)
    f_sub = load_font(22, bold=False)
    f_card_h = load_font(24, bold=True)
    f_body = load_font(18, bold=False)
    f_footer = load_font(16, bold=False)

    draw.rectangle([0, 0, canvas_w, 6], fill=(16, 185, 129))
    draw.text((120, 90), "Summary & Production Readiness", fill=(248, 250, 252), font=f_title)
    draw.text((120, 155), "Key Engineering Takeaways & Real-World Deployment Capabilities", fill=(148, 163, 184), font=f_sub)
    draw.line([(120, 205), (1800, 205)], fill=(51, 65, 85), width=2)

    takeaways = [
        ("01", "Sub-Pixel Edge Matting Precision",
         "The bilinear upsampling decoder effectively eliminated checkerboard artifacts typically generated by standard transpose convolutions. Even fine hair strands, transparent apparel, and complex silhouettes are segmented smoothly without halo borders.", (59, 130, 246)),
        ("02", "Dual BCEDice Loss Convergence",
         "Combining Soft Dice Loss with Binary Cross-Entropy resolved foreground/background class imbalances. The model converged to a 98.39% Dice Score and 96.92% IoU, ensuring reliable boundary classification on diverse backgrounds.", (6, 182, 212)),
        ("03", "Letterbox High-Resolution Scalability",
         "Aspect-ratio preserving letterbox padding allows the compact 256x256 model to process arbitrary multi-megapixel images (tested up to 32.6 MP and 20.3 MP) with zero distortion and minimal memory overhead.", (16, 185, 129)),
        ("04", "Production Ready Pipeline",
         "Fully modular PyTorch architecture supporting Multi-GPU DataParallel inference, transparent RGBA PNG export, and instant solid/studio background compositing. Ready for e-commerce, headshot matting, and creative editing pipelines.", (168, 85, 247)),
    ]

    card_y = 250
    card_w = 780
    card_h = 290
    card_gap_x = 80
    card_gap_y = 50

    for idx, (num, heading, text, accent) in enumerate(takeaways):
        row = idx // 2
        col = idx % 2
        cx = 120 + col * (card_w + card_gap_x)
        cy = card_y + row * (card_h + card_gap_y)

        draw.rounded_rectangle([cx, cy, cx + card_w, cy + card_h], radius=16, fill=(24, 33, 47), outline=(51, 65, 85), width=1)
        draw.rounded_rectangle([cx + 24, cy + 24, cx + 80, cy + 60], radius=8, fill=(30, 41, 59))
        draw.text((cx + 38, cy + 30), num, fill=accent, font=load_font(18, bold=True))
        draw.text((cx + 100, cy + 30), heading, fill=(248, 250, 252), font=f_card_h)

        words = text.split(" ")
        lines = []
        cur_line = []
        for w in words:
            cur_line.append(w)
            bbox = draw.textbbox((0, 0), " ".join(cur_line), font=f_body)
            if bbox[2] - bbox[0] > (card_w - 48):
                cur_line.pop()
                lines.append(" ".join(cur_line))
                cur_line = [w]
        if cur_line:
            lines.append(" ".join(cur_line))

        for l_idx, line in enumerate(lines):
            draw.text((cx + 24, cy + 85 + l_idx * 30), line, fill=(148, 163, 184), font=f_body)

    # Footer
    draw.line([(120, 970), (1800, 970)], fill=(51, 65, 85), width=1)
    draw.text((120, 1000), "Deep Learning Background Removal | GitHub: Razamindset/bg-removal-with-deep-learning", fill=(148, 163, 184), font=f_footer)
    draw.text((1520, 1000), "Built with PyTorch & OpenCV", fill=(56, 189, 248), font=load_font(16, bold=True))

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Export comparison images and LinkedIn showcase PDF")
    parser.add_argument("--input-dir", default="test-images", help="Directory with input test images")
    parser.add_argument("--pred-dir", default="outputs/predictions", help="Directory with predicted transparent PNGs")
    parser.add_argument("--output-dir", default="outputs/comparisons", help="Directory to save comparison outputs")
    parser.add_argument("--curves", default="outputs/curves.png", help="Path to training curves image")
    parser.add_argument("--weights", default="outputs/best_model.pth", help="Model weights checkpoint")
    parser.add_argument("--rerun", action="store_true", help="Re-run inference directly on input images")
    parser.add_argument("--pdf-out", default="outputs/background_removal_showcase.pdf", help="Output path for the compiled PDF")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.pdf_out)), exist_ok=True)

    # Collect input images
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    input_files = []
    for ext in exts:
        input_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
    input_files.sort(key=lambda x: [int(c) if c.isdigit() else c for c in os.path.basename(x).split(".")])

    if not input_files:
        print(f"No input images found in {args.input_dir}")
        return

    print(f"Found {len(input_files)} input images in {args.input_dir}")

    # Model loading if re-running
    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.rerun:
        print(f"Re-running inference with weights from {args.weights} on {device}...")
        model = ResNetUNet(pretrained=False)
        state = torch.load(args.weights, map_location=device)
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(state_dict)
        model.to(device).eval()

    case_titles = [
        "Delicate Edge & Shoulder Silhouette Isolation",
        "Studio Portrait Matting with Complex Attire",
        "Close-Up Portrait Hair Strand Preservation",
        "Low-Resolution Image Detail Recovery",
        "Ultra High-Resolution (32.6 MP) Full-Body Subject",
        "Wide-Angle (20.3 MP) Environmental Landscape Scene",
        "Dynamic Outdoor Framing with Intricate Contours",
        "Subject Segmentation with High Foreground Ratio"
    ]

    all_slides: List[Image.Image] = []

    # 1. Slide 01: Cover Slide
    print("Generating Slide 01: Executive Cover...")
    cover_slide = build_slide_cover()
    all_slides.append(cover_slide)
    cover_slide.save(os.path.join(args.output_dir, "slide_01_cover.png"), quality=95)

    # 2. Slide 02: Training Curves
    print("Generating Slide 02: Training Curves...")
    curves_slide = build_slide_curves(args.curves)
    all_slides.append(curves_slide)
    curves_slide.save(os.path.join(args.output_dir, "slide_02_curves.png"), quality=95)

    # 3. Individual Comparison Images & Case Study Slides
    for idx, in_file in enumerate(input_files, start=1):
        stem = os.path.splitext(os.path.basename(in_file))[0]
        raw_img = Image.open(in_file).convert("RGB")

        # Get RGBA prediction
        pred_file = os.path.join(args.pred_dir, f"{stem}_nobg.png")
        if args.rerun or not os.path.exists(pred_file):
            if model is None:
                print(f"Loading model for missing prediction on {in_file}...")
                model = ResNetUNet(pretrained=False)
                state = torch.load(args.weights, map_location=device)
                state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
                model.load_state_dict(state_dict)
                model.to(device).eval()
            rgba_img = run_inference_on_image(model, raw_img, device)
            os.makedirs(args.pred_dir, exist_ok=True)
            rgba_img.save(pred_file)
        else:
            rgba_img = Image.open(pred_file).convert("RGBA")

        # 3.1 Generate Standalone 3-Panel Comparison Image
        triplet_img = build_compact_triplet(raw_img, rgba_img, f"#{idx} ({stem})")
        triplet_path = os.path.join(args.output_dir, f"{stem}_comparison.png")
        triplet_img.save(triplet_path, quality=95)
        print(f"[{idx}/{len(input_files)}] Saved standalone comparison: {triplet_path}")

        # 3.2 Generate Presentation Slide (1920x1080)
        title_desc = case_titles[(idx - 1) % len(case_titles)]
        slide = build_slide_case_study(raw_img, rgba_img, idx, title_desc)
        slide_path = os.path.join(args.output_dir, f"slide_{idx+2:02d}_case_study_{stem}.png")
        slide.save(slide_path, quality=95)
        all_slides.append(slide)

    # 4. Slide 11: Summary & Production Readiness
    print("Generating Slide 11: Summary & Conclusion...")
    summary_slide = build_slide_summary()
    all_slides.append(summary_slide)
    summary_slide.save(os.path.join(args.output_dir, "slide_11_summary.png"), quality=95)

    # 5. Compile into Single Full PDF
    print(f"\nCompiling all {len(all_slides)} slides into PDF: {args.pdf_out}...")
    all_slides[0].save(
        args.pdf_out,
        save_all=True,
        append_images=all_slides[1:],
        resolution=150.0,
        quality=92,
    )

    pdf_size_mb = os.path.getsize(args.pdf_out) / (1024 * 1024)
    print(f"\nAll operations completed successfully!")
    print(f"-> Individual Comparisons saved to: {args.output_dir}/")
    print(f"-> Full PDF Presentation saved to: {args.pdf_out} ({pdf_size_mb:.2f} MB, {len(all_slides)} pages)")


if __name__ == "__main__":
    main()
