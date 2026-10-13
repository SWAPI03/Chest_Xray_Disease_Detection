"""
models.py
---------
Keras models for chest X-ray pneumonia detection.

Supported backbones:
  - DenseNet121  (primary, mirrors CheXNet, Rajpurkar et al. 2017)
  - ResNet50     (alternative baseline)

Both use ImageNet pretrained weights and a small custom classification head.
Each backbone expects its own input preprocessing, see get_preprocess().
"""

from typing import Callable

import numpy as np
import tensorflow as tf
import keras
from keras import layers


IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406])
IMAGENET_STD  = tf.constant([0.229, 0.224, 0.225])
CAFFE_MEAN_BGR = tf.constant([103.939, 116.779, 123.68])


# ─── Preprocessing ────────────────────────────────────────────────────────────

def _torch_style(x: tf.Tensor) -> tf.Tensor:
    """Scale to [0, 1] and normalise with ImageNet mean/std (DenseNet weights)."""
    return (x / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def _caffe_style(x: tf.Tensor) -> tf.Tensor:
    """RGB -> BGR and subtract the ImageNet channel means (ResNet50 weights)."""
    return tf.reverse(x, axis=[-1]) - CAFFE_MEAN_BGR


# name -> (constructor, preprocessing, Grad-CAM target layer)
BACKBONES = {
    "densenet121": (keras.applications.DenseNet121, _torch_style, "relu"),
    "resnet50":    (keras.applications.ResNet50,    _caffe_style, "conv5_block3_out"),
}


def _lookup(name: str):
    name = name.lower()
    if name not in BACKBONES:
        raise ValueError(f"Unknown model '{name}'. Choose one of {list(BACKBONES)}.")
    return BACKBONES[name]


def get_preprocess(name: str) -> Callable[[tf.Tensor], tf.Tensor]:
    """Preprocessing that matches the backbone's ImageNet weights (input: 0-255 RGB)."""
    return _lookup(name)[1]


def gradcam_layer(name: str) -> str:
    """Name of the last convolutional feature map, used for Grad-CAM."""
    return _lookup(name)[2]


# ─── Model ────────────────────────────────────────────────────────────────────

def build_model(
    name: str = "densenet121",
    image_size: int = 224,
    pretrained: bool = True,
    dropout: float = 0.5,
) -> keras.Model:
    """
    Backbone features → GlobalAvgPool → Dropout → Dense(1, sigmoid).

    Keras' DenseNet121 already ends with BN + ReLU, matching CheXNet's
    features → ReLU → pool → FC head. The output is P(pneumonia).
    The backbone layers sit directly in the model graph (not nested), so
    Grad-CAM can read the last conv feature map by layer name.
    """
    constructor = _lookup(name)[0]
    base = constructor(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=(image_size, image_size, 3),
    )
    x = layers.GlobalAveragePooling2D(name="avg_pool_head")(base.output)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    # float32 output keeps the sigmoid stable under mixed precision
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32", name="pneumonia_prob")(x)
    model = keras.Model(base.input, outputs, name=name.lower())

    total     = model.count_params()
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    print(f"  Model : {name.upper()}  |  Params: {total/1e6:.1f}M  |  Trainable: {trainable/1e6:.1f}M")
    return model


def load_model(path: str) -> keras.Model:
    model = keras.models.load_model(path)
    print(f"  Loaded model: {path}")
    return model
