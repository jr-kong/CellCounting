#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import imageio.v3 as iio
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("imageio>=2.20 is required to read TIFF files.") from exc

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional path
    Image = None


@dataclass(frozen=True)
class PhotometricParams:
    """Parameter bundle for lightweight serialization."""

    brightness_shift: float
    contrast_factor: float
    gamma: float
    noise_std: float

    def asdict(self) -> Dict[str, float]:
        return {
            "brightness_shift": self.brightness_shift,
            "contrast_factor": self.contrast_factor,
            "gamma": self.gamma,
            "noise_std": self.noise_std,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pad, augment, and package microscopy images for cell counting."
    )
    parser.add_argument("--image-dir", type=Path, default=Path("w1_images"))
    parser.add_argument("--mask-dir", type=Path, default=Path("w1_mask"))
    parser.add_argument("--output-dir", type=Path, default=Path("processed"))
    parser.add_argument("--metadata-name", type=str, default="metadata.csv")
    parser.add_argument(
        "--rotations",
        type=int,
        nargs="+",
        default=[0, 90, 180, 270],
        help="Rotation degrees (multiples of 90).",
    )
    parser.add_argument(
        "--photometric-samples",
        type=int,
        default=0,
        help="Extra random photometric variants per rotation.",
    )
    parser.add_argument(
        "--brightness-shift",
        type=float,
        default=0.1,
        help="Uniform brightness shift range in [−range,+range] (after scaling to 0-1).",
    )
    parser.add_argument(
        "--contrast-range",
        type=float,
        default=0.1,
        help="Contrast multiplier draws from [1−range, 1+range].",
    )
    parser.add_argument(
        "--gamma-range",
        type=float,
        default=0.1,
        help="Gamma draws from [1−range, 1+range]; set 0 to disable.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.01,
        help="Additive Gaussian noise std (in 0-1 space).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for photometric augmentation RNG.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=None,
        help="Override square size; otherwise inferred from the largest frame.",
    )
    parser.add_argument(
        "--precision",
        choices=("float32", "float16"),
        default="float32",
        help="Output dtype for stored tensors.",
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=Path("samples"),
        help="Subdirectory of output-dir for per-sample NPZ files.",
    )
    return parser.parse_args()


def list_image_files(directory: Path) -> List[Path]:
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def get_image_shape(path: Path) -> Tuple[int, int]:
    if Image is not None:
        try:
            with Image.open(path) as img:
                return img.height, img.width
        except Exception:
            pass
    array = iio.imread(path)
    if array.ndim == 2:
        return array.shape
    return array.shape[0], array.shape[1]


def load_image_01(path: Path) -> Tuple[np.ndarray, Dict[str, float]]:
    array = iio.imread(path).astype(np.float32)
    if array.ndim == 3:
        array = array[..., 0]
    min_val = float(np.min(array))
    max_val = float(np.max(array))
    if max_val > min_val:
        scaled = (array - min_val) / (max_val - min_val)
    else:
        scaled = np.zeros_like(array, dtype=np.float32)
    return scaled, {"min": min_val, "max": max_val}


def load_mask(path: Path) -> np.ndarray:
    mask = iio.imread(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def count_components(binary: np.ndarray) -> int:
    visited = np.zeros(binary.shape, dtype=bool)
    rows, cols = binary.shape
    count = 0
    stack: List[Tuple[int, int]] = []
    neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    for r in range(rows):
        for c in range(cols):
            if not binary[r, c] or visited[r, c]:
                continue
            count += 1
            stack.append((r, c))
            visited[r, c] = True
            while stack:
                cr, cc = stack.pop()
                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if binary[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
    return count


def pad_to_square(array: np.ndarray, target: int) -> np.ndarray:
    height, width = array.shape
    if height > target or width > target:
        raise ValueError(
            f"Target size {target} too small for frame {height}x{width}. "
            "Increase --target-size."
        )
    pad_vert = target - height
    pad_horiz = target - width
    top = pad_vert // 2
    bottom = pad_vert - top
    left = pad_horiz // 2
    right = pad_horiz - left
    return np.pad(array, ((top, bottom), (left, right)), mode="constant")


def rotate_square(array: np.ndarray, angle: int) -> np.ndarray:
    if angle % 90 != 0:
        raise ValueError("Only right-angle rotations are supported.")
    k = (angle // 90) % 4
    return np.rot90(array, k=k)


def generate_photometric_variants(
    image: np.ndarray,
    rng: np.random.Generator,
    samples: int,
    brightness_shift: float,
    contrast_range: float,
    gamma_range: float,
    noise_std: float,
) -> List[Tuple[str, np.ndarray, PhotometricParams]]:
    variants: List[Tuple[str, np.ndarray, PhotometricParams]] = []
    identity = PhotometricParams(0.0, 1.0, 1.0, 0.0)
    variants.append(("identity", image.copy(), identity))
    if samples <= 0:
        return variants
    for idx in range(samples):
        shift = 0.0
        if brightness_shift > 0:
            shift = float(
                rng.uniform(-brightness_shift, brightness_shift)
            )
        contrast = 1.0
        if contrast_range > 0:
            contrast = float(
                rng.uniform(1.0 - contrast_range, 1.0 + contrast_range)
            )
        gamma = 1.0
        if gamma_range > 0:
            gamma = float(
                rng.uniform(1.0 - gamma_range, 1.0 + gamma_range)
            )
        noise = noise_std if noise_std > 0 else 0.0
        params = PhotometricParams(shift, contrast, gamma, noise)
        augmented = apply_photometric(image, params, rng)
        variants.append((f"rand{idx:02d}", augmented, params))
    return variants


def apply_photometric(
    image: np.ndarray,
    params: PhotometricParams,
    rng: np.random.Generator,
) -> np.ndarray:
    augmented = image.copy()
    if params.contrast_factor != 1.0:
        augmented = ((augmented - 0.5) * params.contrast_factor) + 0.5
    if params.brightness_shift != 0.0:
        augmented = augmented + params.brightness_shift
    augmented = np.clip(augmented, 0.0, 1.0)
    if params.gamma != 1.0:
        augmented = np.clip(augmented, 0.0, 1.0) ** params.gamma
    if params.noise_std > 0.0:
        noise = rng.normal(
            0.0, params.noise_std, size=augmented.shape
        ).astype(np.float32)
        augmented = augmented + noise
    return np.clip(augmented, 0.0, 1.0)


def relative_path(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def main() -> None:
    args = parse_args()

    image_dir = args.image_dir.resolve()
    mask_dir = args.mask_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.sample_dir.is_absolute():
        sample_dir = args.sample_dir.resolve()
    else:
        sample_dir = (output_dir / args.sample_dir).resolve()
    sample_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_image_files(image_dir)
    if not image_paths:
        raise SystemExit(f"No supported images found in {image_dir}.")

    rotations = sorted({angle % 360 for angle in args.rotations})
    for angle in rotations:
        if angle % 90 != 0:
            raise SystemExit("Rotations must be multiples of 90 degrees.")
    if not rotations:
        rotations = [0]

    if args.target_size is not None:
        target_size = args.target_size
    else:
        target_size = 0
        for path in image_paths:
            height, width = get_image_shape(path)
            target_size = max(target_size, height, width)
        if target_size == 0:
            raise SystemExit("Unable to determine target size.")

    dtype = np.float32 if args.precision == "float32" else np.float16
    rng = np.random.default_rng(args.seed)

    total_samples = len(image_paths) * len(rotations) * (
        args.photometric_samples + 1
    )

    common_root = Path(
        os.path.commonpath(
            [str(image_dir), str(mask_dir), str(output_dir), str(sample_dir)]
        )
    )

    metadata_rows: List[Dict[str, object]] = []
    processed = 0

    for image_path in image_paths:
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Missing mask for {image_path.name}: {mask_path}"
            )

        raw_image, stats = load_image_01(image_path)
        padded = pad_to_square(raw_image, target_size)
        mask = load_mask(mask_path)
        cell_count = count_components(mask)
        original_height, original_width = raw_image.shape

        for angle in rotations:
            rotated = rotate_square(padded, angle)
            variants = generate_photometric_variants(
                rotated,
                rng,
                args.photometric_samples,
                args.brightness_shift,
                args.contrast_range,
                args.gamma_range,
                args.noise_std,
            )
            for variant_id, variant_img, params in variants:
                mean = float(np.mean(variant_img))
                std = float(np.std(variant_img))
                denom = std if std > 1e-6 else 1.0
                normalized = (variant_img - mean) / denom
                normalized = normalized.astype(dtype, copy=False)
                normalized = normalized[np.newaxis, ...]

                record_id = f"{image_path.stem}_rot{angle:03d}_{variant_id}"
                sample_path = sample_dir / f"{record_id}.npz"
                np.savez_compressed(
                    sample_path,
                    image=normalized,
                    count=np.int32(cell_count),
                    record_id=record_id,
                    source_image=relative_path(image_path, common_root),
                    source_mask=relative_path(mask_path, common_root),
                    rotation_deg=np.int32(angle),
                    photometric_id=variant_id,
                    photometric_params=json.dumps(params.asdict()),
                    target_size=np.int32(target_size),
                    orig_height=np.int32(original_height),
                    orig_width=np.int32(original_width),
                    dtype=args.precision,
                )
                processed += 1

                row = {
                    "record_id": record_id,
                    "sample_npz": relative_path(sample_path, common_root),
                    "source_image": relative_path(image_path, common_root),
                    "source_mask": relative_path(mask_path, common_root),
                    "orig_height": original_height,
                    "orig_width": original_width,
                    "target_size": target_size,
                    "rotation_deg": angle,
                    "photometric_id": variant_id,
                    "photometric_params": json.dumps(params.asdict()),
                    "cell_count": int(cell_count),
                    "mean_before_norm": mean,
                    "std_before_norm": std,
                    "raw_min": stats["min"],
                    "raw_max": stats["max"],
                    "dtype": args.precision,
                }
                metadata_rows.append(row)

    if processed != total_samples:
        print(
            "Warning: sample count mismatch; proceeding with collected samples."
        )

    metadata_path = output_dir / args.metadata_name

    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "record_id",
                "sample_npz",
                "source_image",
                "source_mask",
                "orig_height",
                "orig_width",
                "target_size",
                "rotation_deg",
                "photometric_id",
                "photometric_params",
                "cell_count",
                "mean_before_norm",
                "std_before_norm",
                "raw_min",
                "raw_max",
                "dtype",
            ],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(
        f"Saved {processed} samples to "
        f"{relative_path(sample_dir, common_root)} "
        f"and metadata to {metadata_path.name}."
    )


if __name__ == "__main__":
    main()
