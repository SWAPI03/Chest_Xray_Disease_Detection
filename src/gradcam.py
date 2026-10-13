"""
gradcam.py
----------
Grad-CAM: highlights the chest X-ray regions that most influenced the prediction.

Keras version of the usual recipe: a second model returns both the last conv
feature map and the prediction, tf.GradientTape gives the gradient of the class
score with respect to that feature map, and the pooled gradients weight it.
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from src.dataset import CLASSES, load_image


class GradCAM:
    def __init__(self, model: keras.Model, layer_name: str, preprocess: Callable):
        self.preprocess = preprocess
        self.grad_model = keras.Model(
            model.inputs, [model.get_layer(layer_name).output, model.output],
        )

    def generate(
        self,
        image: np.ndarray,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        image: (H, W, 3) uint8 RGB as returned by dataset.load_image.
        Returns (heatmap in [0, 1], overlay RGB uint8, P(pneumonia)).
        """
        x = self.preprocess(tf.convert_to_tensor(image[None], tf.float32))

        with tf.GradientTape() as tape:
            conv_out, prob = self.grad_model(x, training=False)
            p = tf.cast(prob[:, 0], tf.float32)
            if target_class is None:
                target_class = int(p[0] >= 0.5)
            score = p if target_class == 1 else 1.0 - p

        # Differentiate w.r.t. the model's own tensor, then cast: under mixed
        # precision the feature map is float16.
        grads    = tf.cast(tape.gradient(score, conv_out), tf.float32)   # (1, h, w, C)
        conv_out = tf.cast(conv_out, tf.float32)
        weights  = tf.reduce_mean(grads, axis=(1, 2))                    # (1, C)
        cam     = tf.nn.relu(tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1))[0]
        cam     = cam.numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        heatmap = cv2.resize(cam, (image.shape[1], image.shape[0]))
        return heatmap, overlay(image, heatmap), float(p[0])


def overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a jet-coloured heatmap over an RGB image."""
    heat = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 1 - alpha, heat, alpha, 0)


def save_comparison(
    image: np.ndarray,
    heatmap: np.ndarray,
    overlay_img: np.ndarray,
    prediction: str,
    confidence: float,
    output_path: str,
    title_prefix: str = "",
):
    """Save original | heatmap | overlay side-by-side."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f"{title_prefix}Grad-CAM  |  Prediction: {prediction}  |  Confidence: {confidence:.1%}",
        fontsize=13, fontweight="bold",
    )
    axes[0].imshow(image)
    axes[0].set_title("Original X-Ray", fontweight="bold")
    im = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Activation Heatmap", fontweight="bold")
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    axes[2].imshow(overlay_img)
    axes[2].set_title("Grad-CAM Overlay", fontweight="bold")
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved Grad-CAM: {output_path}")


def save_gradcam_examples(
    model: keras.Model,
    layer_name: str,
    preprocess: Callable,
    paths: List[str],
    labels,
    probs,
    output_dir: str,
    image_size: int = 224,
    threshold: float = 0.5,
    n: int = 4,
):
    """
    Grad-CAM on real test images:
      gradcam_example.png   one confidently detected pneumonia case (3 panels)
      gradcam_examples.png  n cases: correct pneumonia, correct normal, and a mistake if any
    """
    labels, probs = np.asarray(labels), np.asarray(probs)
    preds = (probs >= threshold).astype(int)
    cam = GradCAM(model, layer_name, preprocess)

    tp = np.flatnonzero((labels == 1) & (preds == 1))
    tn = np.flatnonzero((labels == 0) & (preds == 0))
    wrong = np.flatnonzero(labels != preds)

    # Most confident true positive for the single headline example
    if len(tp):
        i = tp[np.argmax(probs[tp])]
        img = load_image(paths[i], image_size)
        heat, over, p = cam.generate(img, target_class=1)
        save_comparison(img, heat, over, "Pneumonia", p,
                        str(Path(output_dir) / "gradcam_example.png"))

    rng = np.random.default_rng(7)
    picks  = list(rng.choice(tp, min(2, len(tp)), replace=False)) if len(tp) else []
    picks += list(rng.choice(tn, min(1, len(tn)), replace=False)) if len(tn) else []
    picks += list(rng.choice(wrong, min(1, len(wrong)), replace=False)) if len(wrong) else []
    picks = picks[:n]
    if not picks:
        return

    fig, axes = plt.subplots(len(picks), 2, figsize=(8, 4 * len(picks)))
    axes = np.atleast_2d(axes)
    for row, i in zip(axes, picks):
        img = load_image(paths[i], image_size)
        heat, over, p = cam.generate(img, target_class=int(preds[i]))
        conf = p if preds[i] == 1 else 1 - p
        ok = preds[i] == labels[i]
        row[0].imshow(img)
        row[0].set_title(f"True: {CLASSES[labels[i]]}", fontweight="bold")
        row[1].imshow(over)
        row[1].set_title(f"Pred: {CLASSES[preds[i]]} ({conf:.1%})",
                         color="#2E7D32" if ok else "#C62828", fontweight="bold")
        for ax in row:
            ax.axis("off")
    fig.suptitle("Grad-CAM on test images", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = Path(output_dir) / "gradcam_examples.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
