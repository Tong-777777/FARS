import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_image_file(path):
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def list_images(folder, sample_limit=None, seed=42):
    paths = sorted(path for path in Path(folder).iterdir() if is_image_file(path))
    if sample_limit is not None and len(paths) > sample_limit:
        rng = random.Random(seed)
        paths = sorted(rng.sample(paths, sample_limit))
    return paths


def load_rgb(path, max_size=512):
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def rgb_to_gray(img):
    return img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114


def box_blur_gray(gray, kernel_size):
    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1

    pad = kernel_size // 2
    padded = np.pad(gray, pad_width=pad, mode="reflect")
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")

    h, w = gray.shape
    y0 = np.arange(h)
    y1 = y0 + kernel_size
    x0 = np.arange(w)
    x1 = x0 + kernel_size

    area = kernel_size * kernel_size
    return (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    ) / area


def laplacian_variance(gray):
    padded = np.pad(gray, pad_width=1, mode="reflect")
    lap = (
        -4.0 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return float(np.var(lap))


def image_stats(path, max_size):
    img = load_rgb(path, max_size=max_size)
    gray = rgb_to_gray(img)
    h, w = gray.shape
    min_side = min(h, w)

    low_kernel = max(9, min_side // 4)
    high_kernel = max(3, min_side // 32)

    low = box_blur_gray(gray, low_kernel)
    base = box_blur_gray(gray, high_kernel)
    high = gray - base

    mean = float(gray.mean())
    p10, p20, p50, p80, p90 = np.quantile(gray, [0.10, 0.20, 0.50, 0.80, 0.90])
    contrast = float(p90 - p10)
    low_texture = float(np.std(low - low.mean()))
    high_noise = float(np.std(high))
    sharpness = laplacian_variance(gray)

    dark_threshold = min(p20, mean - gray.std())
    dark_area = float((gray < dark_threshold).mean())
    dark_factor = float(max(p10, 1e-4) / max(p50, 1e-4))

    return {
        "mean": mean,
        "contrast": contrast,
        "low_texture": low_texture,
        "high_noise": high_noise,
        "sharpness": sharpness,
        "dark_area": dark_area,
        "dark_factor": dark_factor,
    }


def collect_stats(paths, max_size):
    if not paths:
        raise RuntimeError("No images found.")

    stats = [image_stats(path, max_size=max_size) for path in paths]
    return {key: np.asarray([item[key] for item in stats], dtype=np.float32) for key in stats[0]}


def q(values, lo=0.10, hi=0.90):
    return np.quantile(values, [lo, hi]).astype(float)


def clamp_pair(values, min_value, max_value, min_width=1e-4):
    lo, hi = float(values[0]), float(values[1])
    lo = max(min_value, min(max_value, lo))
    hi = max(min_value, min(max_value, hi))
    if hi < lo + min_width:
        hi = min(max_value, lo + min_width)
    return [lo, hi]


def round_list(values, digits=4):
    return [round(float(value), digits) for value in values]


def calibrate(flat_stats, rough_stats):
    flat_mean_med = float(np.median(flat_stats["mean"]))
    flat_contrast_med = float(np.median(flat_stats["contrast"])) + 1e-6
    flat_noise_med = float(np.median(flat_stats["high_noise"]))
    flat_sharpness_med = float(np.median(flat_stats["sharpness"])) + 1e-8

    rough_mean_q = q(rough_stats["mean"], 0.10, 0.90)
    rough_contrast_q = q(rough_stats["contrast"], 0.10, 0.90)
    rough_low_texture_q = q(rough_stats["low_texture"], 0.10, 0.90)
    rough_noise_extra_q = q(np.maximum(rough_stats["high_noise"] - flat_noise_med, 0.0), 0.10, 0.90)

    contrast_range = clamp_pair(rough_contrast_q / flat_contrast_med, 0.45, 1.80)
    brightness_range = clamp_pair(rough_mean_q - flat_mean_med, -0.30, 0.30)

    flat_mean_safe = np.clip(flat_mean_med, 0.05, 0.95)
    rough_mean_safe = np.clip(rough_mean_q, 0.05, 0.95)
    gamma_range = clamp_pair(np.log(rough_mean_safe) / np.log(flat_mean_safe), 0.50, 1.80)

    texture_strength_range = clamp_pair(rough_low_texture_q, 0.00, 0.18)
    noise_std_range = clamp_pair(rough_noise_extra_q, 0.00, 0.10)

    shadow_prob = float(np.mean(rough_stats["dark_area"] > 0.03))
    shadow_prob = round(float(np.clip(shadow_prob, 0.10, 0.80)), 3)
    shadow_dark_factor_range = clamp_pair(q(rough_stats["dark_factor"], 0.10, 0.90), 0.35, 0.90)

    sharpness_ratio = rough_stats["sharpness"] / flat_sharpness_med
    blur_prob = float(np.clip(np.mean(sharpness_ratio < 0.80) * 0.45, 0.05, 0.35))
    sharpen_prob = float(np.clip(np.mean(sharpness_ratio > 1.20) * 0.45, 0.05, 0.35))

    return {
        "use_rough_style_aug": True,
        "rough_style_prob": 0.7,
        "contrast_range": round_list(contrast_range),
        "brightness_range": round_list(brightness_range),
        "gamma_range": round_list(gamma_range),
        "noise_std_range": round_list(noise_std_range),
        "texture_strength_range": round_list(texture_strength_range),
        "shadow_prob": shadow_prob,
        "shadow_rx_ratio_range": [0.10, 0.45],
        "shadow_ry_ratio_range": [0.03, 0.18],
        "shadow_dark_factor_range": round_list(shadow_dark_factor_range),
        "blur_prob": round(blur_prob, 3),
        "sharpen_prob": round(sharpen_prob, 3),
    }


def print_yaml_block(stage3):
    print("stage3:")
    for key, value in stage3.items():
        if isinstance(value, list):
            print(f"  {key}: [{value[0]}, {value[1]}]")
        elif isinstance(value, bool):
            print(f"  {key}: {str(value)}")
        else:
            print(f"  {key}: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate FARS augmentation ranges from real rough images."
    )
    parser.add_argument("--flat_dir", default=r'', help="Directory of labeled flat images.")
    parser.add_argument("--rough_dir", default=r'', help="Directory of real rough images.")
    parser.add_argument("--sample_limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_size", type=int, default=1024)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    flat_paths = list_images(args.flat_dir, sample_limit=args.sample_limit, seed=args.seed)
    rough_paths = list_images(args.rough_dir, sample_limit=args.sample_limit, seed=args.seed)

    print(f"# flat images: {len(flat_paths)}")
    print(f"# rough images: {len(rough_paths)}")

    flat_stats = collect_stats(flat_paths, max_size=args.max_size)
    rough_stats = collect_stats(rough_paths, max_size=args.max_size)
    stage3_cfg = calibrate(flat_stats, rough_stats)

    print()
    print_yaml_block(stage3_cfg)
