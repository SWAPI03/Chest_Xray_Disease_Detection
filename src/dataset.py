"""
dataset.py
----------
Dataset and DataLoader for the Kaggle Chest X-Ray (Pneumonia) dataset.

Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
Classes: NORMAL (0) | PNEUMONIA (1)

Expected folder layout after download and unzip:
    data/raw/
        train/
            NORMAL/     *.jpeg
            PNEUMONIA/  *.jpeg
        val/
            NORMAL/
            PNEUMONIA/
        test/
            NORMAL/
            PNEUMONIA/
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─── Constants ────────────────────────────────────────────────────────────────

CLASSES      = ["Normal", "Pneumonia"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─── Augmentation pipelines ───────────────────────────────────────────────────

def get_train_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.2),
        A.CLAHE(clip_limit=2.0, p=0.3),   # boosts lung detail contrast
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ─── Dataset ──────────────────────────────────────────────────────────────────

class ChestXrayDataset(Dataset):
    """
    Loads NORMAL / PNEUMONIA images from a split folder.

    Args:
        root_dir:   Path to data/raw/
        split:      'train' | 'val' | 'test'
        image_size: Resize target (default 224)
        transform:  Optional albumentations pipeline override
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 224,
        transform: Optional[A.Compose] = None,
    ):
        self.split_dir = Path(root_dir) / split
        self.transform = transform or (
            get_train_transforms(image_size) if split == "train"
            else get_val_transforms(image_size)
        )
        self.samples: List[Tuple[str, int]] = []
        self._load()

    def _load(self):
        for label, folder in enumerate(["NORMAL", "PNEUMONIA"]):
            folder_path = self.split_dir / folder
            if not folder_path.exists():
                raise FileNotFoundError(
                    f"Folder not found: {folder_path}\n"
                    f"Make sure you've downloaded the dataset to data/raw/\n"
                    f"See README.md → Quick Start → Step 2."
                )
            for p in folder_path.glob("*"):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((str(p), label))

        n_normal    = sum(1 for _, l in self.samples if l == 0)
        n_pneumonia = sum(1 for _, l in self.samples if l == 1)
        print(f"  [{self.split_dir.name}]  Normal: {n_normal}  |  Pneumonia: {n_pneumonia}  |  Total: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        path, label = self.samples[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image)
        return {
            "image": augmented["image"],
            "label": torch.tensor(label, dtype=torch.long),
            "path":  path,
        }

    # ── helpers ──────────────────────────────────────────────────────────────

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for CrossEntropyLoss."""
        labels = np.array([l for _, l in self.samples])
        counts = np.bincount(labels).astype(float)
        weights = 1.0 / counts
        return torch.tensor(weights / weights.sum(), dtype=torch.float32)

    def get_sampler(self) -> WeightedRandomSampler:
        """WeightedRandomSampler to oversample the minority class."""
        labels = [l for _, l in self.samples]
        counts = np.bincount(labels)
        sample_weights = [1.0 / counts[l] for l in labels]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )


# ─── DataLoader factory ───────────────────────────────────────────────────────

def get_dataloaders(
    root_dir: str = "data/raw",
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """
    Returns {'train', 'val', 'test'} DataLoaders for the Kaggle dataset.

    Training loader uses WeightedRandomSampler to handle the 3:1 class imbalance.
    """
    train_ds = ChestXrayDataset(root_dir, split="train", image_size=image_size)
    val_ds   = ChestXrayDataset(root_dir, split="val",   image_size=image_size)
    test_ds  = ChestXrayDataset(root_dir, split="test",  image_size=image_size)

    return {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_ds.get_sampler(),   # balanced sampling
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }
