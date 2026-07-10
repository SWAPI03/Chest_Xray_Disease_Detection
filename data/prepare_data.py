"""
data/prepare_data.py
--------------------
Data preparation for the Kaggle Chest X-Ray (Pneumonia) dataset.

Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

After running `kaggle datasets download` the zip extracts to:
    chest_xray/
        train/  NORMAL/  PNEUMONIA/
        val/    NORMAL/  PNEUMONIA/
        test/   NORMAL/  PNEUMONIA/

This script:
  1. Verifies the folder structure is correct
  2. Creates a validation split from training data if val/ is too small
     (the Kaggle val/ folder only has 16 images — we expand it to 10%)

Usage:
    python data/prepare_data.py                  # verify + fix val split
    python data/prepare_data.py --sample         # generate synthetic data for testing
"""

import shutil
import argparse
import numpy as np
import cv2
from pathlib import Path


RAW_DIR = Path("data/raw")


# ─── Verify ───────────────────────────────────────────────────────────────────

def verify(root: Path = RAW_DIR) -> bool:
    """Print a summary of the dataset and check all folders exist."""
    ok = True
    print("\nDataset summary:")
    for split in ["train", "val", "test"]:
        for cls in ["NORMAL", "PNEUMONIA"]:
            folder = root / split / cls
            if not folder.exists():
                print(f"  MISSING  {folder}")
                ok = False
            else:
                n = len(list(folder.glob("*.*")))
                print(f"  {split:5s} / {cls:<10s}: {n:>5} images")
    return ok


# ─── Fix val split ────────────────────────────────────────────────────────────

def fix_val_split(root: Path = RAW_DIR, val_ratio: float = 0.10, seed: int = 42):
    """
    The Kaggle val/ folder has only 16 images (8 per class) — too small.
    This moves val_ratio of training images into val/ to create a proper split.

    Safe to re-run: skips if val/ already has >16 images per class.
    """
    np.random.seed(seed)

    for cls in ["NORMAL", "PNEUMONIA"]:
        val_cls = root / "val" / cls
        n_current = len(list(val_cls.glob("*.*"))) if val_cls.exists() else 0

        if n_current > 16:
            print(f"  {cls}: val already has {n_current} images — skipping")
            continue

        train_cls = root / "train" / cls
        val_cls.mkdir(parents=True, exist_ok=True)

        images = [p for p in train_cls.glob("*.*")
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        n_move = int(len(images) * val_ratio)
        to_move = np.random.choice(images, n_move, replace=False)

        for img in to_move:
            shutil.move(str(img), str(val_cls / img.name))

        print(f"  {cls}: moved {n_move} images  train → val")


# ─── Synthetic sample data ────────────────────────────────────────────────────

def create_sample(output_dir: str = "data/sample", n_per_class: int = 30, size: int = 224):
    """
    Generate synthetic grayscale X-ray-like images for pipeline testing
    without downloading the real dataset.

    Creates: data/sample/train|val|test / NORMAL|PNEUMONIA / *.jpeg
    """
    root = Path(output_dir)
    print(f"\nGenerating synthetic sample data → {root}/")

    splits = {"train": n_per_class, "val": max(5, n_per_class // 5), "test": max(5, n_per_class // 5)}

    for split, n in splits.items():
        for cls, base_val in [("NORMAL", 185), ("PNEUMONIA", 130)]:
            folder = root / split / cls
            folder.mkdir(parents=True, exist_ok=True)

            for i in range(n):
                img = np.random.normal(base_val, 20, (size, size)).clip(0, 255).astype(np.uint8)

                # Simulate lung fields
                for cx in [size // 3, 2 * size // 3]:
                    Y, X = np.ogrid[:size, :size]
                    mask = (X - cx) ** 2 + (Y - size // 2) ** 2 <= (size // 4) ** 2
                    img[mask] = np.clip(img[mask] * 0.7 + 180 * 0.3, 0, 255).astype(np.uint8)

                if cls == "PNEUMONIA":
                    # Add consolidation patch (opacity)
                    cx, cy = size // 3, size // 2
                    Y, X = np.ogrid[:size, :size]
                    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= (size // 8) ** 2
                    img[mask] = np.clip(img[mask] * 0.4 + 80 * 0.6, 0, 255).astype(np.uint8)

                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(str(folder / f"{cls.lower()}_{i:04d}.jpeg"), img_rgb)

    print(f"Done. Use  python main.py --mode train  with config root_dir: {output_dir}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",   default="data/raw",    help="Dataset root directory")
    parser.add_argument("--sample", action="store_true",   help="Generate synthetic test data instead")
    parser.add_argument("--n",      type=int, default=30,  help="Images per class for --sample")
    args = parser.parse_args()

    if args.sample:
        create_sample("data/sample", n_per_class=args.n)
    else:
        root = Path(args.root)
        if verify(root):
            print("\nFixing val split (Kaggle val/ is too small)…")
            fix_val_split(root)
            print("\nFinal dataset:")
            verify(root)
        else:
            print("\nDownload the dataset first:")
            print("  pip install kaggle")
            print("  kaggle datasets download -d paultimothymooney/chest-xray-pneumonia")
            print("  unzip chest-xray-pneumonia.zip -d data/raw/")
