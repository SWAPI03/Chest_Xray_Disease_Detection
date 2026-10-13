"""
train.py
--------
Training for chest X-ray pneumonia detection with Keras model.fit.

Features:
  - Mixed-precision training on GPU (float16 compute, float32 output)
  - AdamW with cosine / step / plateau learning-rate schedules
  - Class weights for the 3:1 Pneumonia:Normal imbalance
  - Early stopping and best-model checkpointing on validation AUC
  - CSV + TensorBoard logging and a training-curves plot
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras

from src.dataset import class_weights

logger = logging.getLogger(__name__)


# ─── Learning-rate schedule ───────────────────────────────────────────────────

def _learning_rate(config: dict, steps_per_epoch: int):
    """Returns (learning_rate, extra_callbacks) for the configured scheduler."""
    cfg_t = config["training"]
    lr, epochs, name = cfg_t["learning_rate"], cfg_t["epochs"], cfg_t["scheduler"]

    if name == "cosine":
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=epochs * steps_per_epoch,
            alpha=1e-6 / lr,
        ), []
    if name == "step":
        return keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[10 * steps_per_epoch, 20 * steps_per_epoch],
            values=[lr, lr * 0.1, lr * 0.01],
        ), []
    if name == "plateau":
        return lr, [keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max", patience=3, factor=0.5, min_lr=1e-6, verbose=1,
        )]
    raise ValueError(f"Unknown scheduler: {name}")


def compile_model(model: keras.Model, config: dict, steps_per_epoch: int) -> list:
    cfg_t = config["training"]
    learning_rate, lr_callbacks = _learning_rate(config, steps_per_epoch)
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=cfg_t["weight_decay"],
            clipnorm=1.0,
        ),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return lr_callbacks


# ─── Main training function ───────────────────────────────────────────────────

def train(model: keras.Model, data: Dict[str, dict], config: dict) -> Tuple[keras.Model, dict]:
    """
    Train the model and return (best model, history).

    Saves the best model (highest validation AUC) to outputs/models/best_model.keras,
    logs to outputs/logs/, and plots outputs/visualizations/training_curves.png.
    """
    cfg_t = config["training"]
    out   = config["output"]
    model_dir, log_dir, viz_dir = (Path(out[k]) for k in ("model_dir", "log_dir", "viz_dir"))
    for d in (model_dir, log_dir, viz_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = data["train"]["ds"], data["val"]["ds"]
    steps_per_epoch = int(train_ds.cardinality())
    lr_callbacks = compile_model(model, config, steps_per_epoch)

    cw = class_weights(data["train"]["labels"]) if cfg_t.get("use_class_weights", True) else None
    best_path = model_dir / out["best_model_name"]

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(best_path), monitor="val_auc", mode="max", save_best_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=cfg_t["patience"], min_delta=1e-4, verbose=1,
        ),
        keras.callbacks.CSVLogger(str(log_dir / "history.csv")),
        keras.callbacks.TensorBoard(log_dir=str(log_dir / "tensorboard")),
        *lr_callbacks,
    ]

    print(f"\n{'='*55}")
    print(f"  Model   : {config['model']['name'].upper()}")
    print(f"  Epochs  : {cfg_t['epochs']}  |  LR: {cfg_t['learning_rate']}  |  Steps/epoch: {steps_per_epoch}")
    print(f"  Policy  : {keras.config.dtype_policy().name}")
    print(f"  Class weights: {cw}")
    print(f"{'='*55}\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg_t["epochs"],
        class_weight=cw,
        callbacks=callbacks,
        verbose=2,
    ).history

    best_epoch = int(np.argmax(history["val_auc"])) + 1
    summary = {
        "epochs_trained": len(history["loss"]),
        "best_epoch": best_epoch,
        "best_val_auc": float(max(history["val_auc"])),
        "class_weights": cw,
        "history": {k: [float(v) for v in vals] for k, vals in history.items()},
    }
    with open(log_dir / "history.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_training_curves(history, best_epoch, viz_dir / "training_curves.png")
    print(f"\nTraining done.  Best Val AUC: {summary['best_val_auc']:.4f} (epoch {best_epoch})  →  {best_path}")

    best = keras.models.load_model(best_path)
    return best, summary


# ─── Training curves ──────────────────────────────────────────────────────────

def plot_training_curves(history: dict, best_epoch: int, out_path: Path):
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("Training history: DenseNet121 (Keras)", fontsize=13, fontweight="bold")

    for ax, key, title in zip(axes, ["loss", "auc", "accuracy"], ["Loss", "AUC-ROC", "Accuracy"]):
        ax.plot(epochs, history[key], "o-", lw=2, ms=4, label="Train")
        ax.plot(epochs, history[f"val_{key}"], "s-", lw=2, ms=4, label="Validation")
        ax.axvline(best_epoch, color="#F44336", ls="--", lw=1.2, label=f"Best epoch ({best_epoch})")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
