"""
dataset.py
----------
tf.data input pipeline for the Kaggle Chest X-Ray (Pneumonia) dataset.

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

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import tensorflow as tf


# ─── Constants ────────────────────────────────────────────────────────────────

CLASSES     = ["Normal", "Pneumonia"]
CLASS_DIRS  = ["NORMAL", "PNEUMONIA"]
IMAGE_EXTS  = {".jpg", ".jpeg", ".png"}
AUTOTUNE    = tf.data.AUTOTUNE


# ─── File listing ─────────────────────────────────────────────────────────────

def list_split(root_dir: str, split: str) -> Tuple[List[str], np.ndarray]:
    """Return sorted image paths and integer labels for one split."""
    split_dir = Path(root_dir) / split
    paths, labels = [], []
    for label, folder in enumerate(CLASS_DIRS):
        folder_path = split_dir / folder
        if not folder_path.exists():
            raise FileNotFoundError(
                f"Folder not found: {folder_path}\n"
                f"Make sure you've downloaded the dataset to data/raw/\n"
                f"See README.md → Setup."
            )
        files = sorted(p for p in folder_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        paths.extend(str(p) for p in files)
        labels.extend([label] * len(files))

    labels = np.array(labels, dtype=np.int32)
    n_normal, n_pneumonia = int((labels == 0).sum()), int((labels == 1).sum())
    print(f"  [{split}]  Normal: {n_normal}  |  Pneumonia: {n_pneumonia}  |  Total: {len(paths)}")
    return paths, labels


def class_weights(labels: np.ndarray) -> Dict[int, float]:
    """Inverse-frequency weights for model.fit(class_weight=...), mean weight = 1."""
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(float)
    weights = len(labels) / (len(CLASSES) * counts)
    return {i: float(w) for i, w in enumerate(weights)}


# ─── Decoding ─────────────────────────────────────────────────────────────────

def _decode(path: tf.Tensor, image_size: int) -> tf.Tensor:
    """Read a file and return a (H, W, 3) uint8 image.

    X-rays are mostly single channel; channels=3 repeats the gray channel so
    the ImageNet-pretrained backbone gets the 3 channels it expects.
    """
    image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    image = tf.image.resize(image, [image_size, image_size])
    return tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)


def load_image(path: str, image_size: int = 224) -> np.ndarray:
    """Load one image exactly like the input pipeline does (uint8 RGB)."""
    return _decode(tf.constant(path), image_size).numpy()


# ─── Augmentation ─────────────────────────────────────────────────────────────
# Plain tf ops on (B, H, W, 3) float batches in the 0-255 range. Every random
# draw is per image, and stateful tf.random ops give fresh values each batch.

def _coin(n: tf.Tensor, p: float) -> tf.Tensor:
    """(n, 1, 1, 1) float mask that is 1 with probability p."""
    return tf.cast(tf.random.uniform([n, 1, 1, 1]) < p, tf.float32)


def _random_flip(images: tf.Tensor) -> tf.Tensor:
    flip = _coin(tf.shape(images)[0], 0.5)
    return flip * tf.reverse(images, axis=[2]) + (1 - flip) * images


def _random_rotate(images: tf.Tensor, max_degrees: float = 10.0, p: float = 0.5) -> tf.Tensor:
    n = tf.shape(images)[0]
    h = tf.cast(tf.shape(images)[1], tf.float32)
    w = tf.cast(tf.shape(images)[2], tf.float32)
    max_rad = max_degrees * np.pi / 180
    angles = tf.random.uniform([n], -max_rad, max_rad)
    angles *= tf.reshape(_coin(n, p), [n])
    cos, sin = tf.cos(angles), tf.sin(angles)
    # Projective transform that rotates about the image centre (output -> input).
    x_off = ((w - 1) - (cos * (w - 1) - sin * (h - 1))) / 2
    y_off = ((h - 1) - (sin * (w - 1) + cos * (h - 1))) / 2
    zeros = tf.zeros_like(cos)
    transforms = tf.stack([cos, -sin, x_off, sin, cos, y_off, zeros, zeros], axis=1)
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=images, transforms=transforms,
        output_shape=tf.shape(images)[1:3], fill_value=0.0,
        interpolation="BILINEAR", fill_mode="REFLECT",
    )


def _random_brightness_contrast(images: tf.Tensor, limit: float = 0.15, p: float = 0.4) -> tf.Tensor:
    n = tf.shape(images)[0]
    apply = _coin(n, p)
    delta  = tf.random.uniform([n, 1, 1, 1], -limit, limit) * 255.0
    factor = tf.random.uniform([n, 1, 1, 1], 1 - limit, 1 + limit)
    mean = tf.reduce_mean(images, axis=[1, 2, 3], keepdims=True)
    jittered = (images - mean) * factor + mean + delta
    return apply * jittered + (1 - apply) * images


def _gaussian_noise(images: tf.Tensor, p: float = 0.2) -> tf.Tensor:
    n = tf.shape(images)[0]
    std = tf.random.uniform([n, 1, 1, 1], 2.2, 5.5)
    return images + _coin(n, p) * std * tf.random.normal(tf.shape(images))


def augment(images: tf.Tensor) -> tf.Tensor:
    """Horizontal flip, ±10° rotation, brightness/contrast jitter, light noise."""
    images = _random_flip(images)
    images = _random_rotate(images)
    images = _random_brightness_contrast(images)
    images = _gaussian_noise(images)
    return tf.clip_by_value(images, 0.0, 255.0)


# ─── Dataset factory ──────────────────────────────────────────────────────────

def make_dataset(
    paths: List[str],
    labels: np.ndarray,
    image_size: int,
    batch_size: int,
    preprocess: Callable[[tf.Tensor], tf.Tensor],
    training: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    paths -> decoded, resized images (cached) -> [shuffle + augment] -> preprocess.

    Decoded images are cached as uint8 after the first epoch, so later epochs
    skip JPEG decoding. Augmentation runs on whole batches after the cache.
    """
    y = labels.astype(np.float32).reshape(-1, 1)
    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    ds = ds.map(lambda p, t: (_decode(p, image_size), t), num_parallel_calls=AUTOTUNE)
    ds = ds.cache()

    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    # Dropping the last partial training batch keeps one input shape, so the
    # XLA-compiled train step isn't compiled a second time.
    ds = ds.batch(batch_size, drop_remainder=training)
    ds = ds.map(lambda x, t: (tf.cast(x, tf.float32), t), num_parallel_calls=AUTOTUNE)

    if training:
        shape = [None, image_size, image_size, 3]  # the rotation op drops the static shape
        ds = ds.map(lambda x, t: (tf.ensure_shape(augment(x), shape), t), num_parallel_calls=AUTOTUNE)

    ds = ds.map(lambda x, t: (preprocess(x), t), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def get_datasets(
    root_dir: str,
    image_size: int,
    batch_size: int,
    preprocess: Callable[[tf.Tensor], tf.Tensor],
    seed: int = 42,
) -> Dict[str, dict]:
    """
    Returns {'train', 'val', 'test'}, each {'ds', 'paths', 'labels'}.

    The class imbalance (about 3:1 Pneumonia:Normal) is handled with
    class weights in model.fit, see class_weights().
    """
    out = {}
    for split in ["train", "val", "test"]:
        paths, labels = list_split(root_dir, split)
        out[split] = {
            "ds": make_dataset(paths, labels, image_size, batch_size, preprocess,
                               training=(split == "train"), seed=seed),
            "paths": paths,
            "labels": labels,
        }
    return out
