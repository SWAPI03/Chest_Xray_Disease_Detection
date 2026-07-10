"""
evaluate.py
-----------
Evaluation metrics and visualizations for pneumonia detection.

Metrics: Accuracy, Precision, Recall, F1, AUC-ROC
Plots:   Confusion Matrix, ROC Curve
Saves:   outputs/reports/metrics.json
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    classification_report,
)
from tqdm import tqdm

CLASSES = ["Normal", "Pneumonia"]


# ─── Core metrics ─────────────────────────────────────────────────────────────

def compute_metrics(
    labels: List[int],
    preds:  List[int],
    probs:  List[float],
) -> Dict[str, float]:
    """Return accuracy / precision / recall / f1 / auc."""
    labels = np.array(labels)
    preds  = np.array(preds)
    probs  = np.array(probs)
    return {
        "accuracy":  float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "auc":       float(roc_auc_score(labels, probs)),
    }


# ─── Full test-set evaluation ────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Optional[str] = None,
    reports_dir: Optional[str] = None,
) -> Dict[str, float]:
    """
    Run the model on the test set, print a report, save plots and metrics.json.

    Args:
        model:       Trained model (eval mode set inside)
        loader:      Test DataLoader
        device:      CPU or CUDA
        output_dir:  Where to save confusion_matrix.png and roc_curve.png
        reports_dir: Where to save metrics.json

    Returns:
        Metrics dict
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for batch in tqdm(loader, desc="  Evaluating"):
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        preds  = logits.argmax(dim=1)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["label"].numpy())

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    _print_report(all_labels, all_preds, metrics)

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        _plot_confusion_matrix(all_labels, all_preds, output_dir)
        _plot_roc_curve(all_labels, all_probs, metrics["auc"], output_dir)

    if reports_dir:
        _save_metrics_json(metrics, reports_dir)

    return metrics


# ─── Console report ───────────────────────────────────────────────────────────

def _print_report(labels, preds, metrics):
    print("\n" + "=" * 50)
    print("  TEST SET RESULTS")
    print("=" * 50)
    print(classification_report(labels, preds, target_names=CLASSES))
    print(f"  AUC-ROC  : {metrics['auc']:.4f}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1 Score : {metrics['f1']:.4f}")
    print("=" * 50 + "\n")


# ─── Confusion matrix ─────────────────────────────────────────────────────────

def _plot_confusion_matrix(labels, preds, output_dir: str):
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Confusion Matrix — DenseNet121  (Test Set, n=624)",
                 fontsize=12, fontweight="bold")

    for ax, data, title, fmt, cmap in zip(
        axes,
        [cm, cm_norm],
        ["Counts", "Normalised"],
        ["d", ".2f"],
        ["Blues", "Greens"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap=cmap,
            xticklabels=CLASSES, yticklabels=CLASSES,
            ax=ax, linewidths=1.5, linecolor="white",
            annot_kws={"size": 14, "weight": "bold"},
        )
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual",    fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")

    plt.tight_layout()
    out = Path(output_dir) / "confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─── ROC curve ────────────────────────────────────────────────────────────────

def _plot_roc_curve(labels, probs, auc: float, output_dir: str):
    fpr, tpr, thresholds = roc_curve(labels, probs)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2.5,
            label=f"DenseNet121  (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.10, color="#2196F3")

    # Mark threshold=0.5 operating point
    idx = np.argmin(np.abs(thresholds - 0.5))
    ax.scatter(fpr[idx], tpr[idx], color="#F44336", s=100, zorder=5,
               label=f"Threshold 0.5  (FPR={fpr[idx]:.2f}, TPR={tpr[idx]:.2f})")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.03])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curve — Pneumonia Detection\n(Test Set n=624)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = Path(output_dir) / "roc_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─── Save metrics.json ────────────────────────────────────────────────────────

def _save_metrics_json(metrics: Dict[str, float], reports_dir: str):
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    out = Path(reports_dir) / "metrics.json"
    with open(out, "w") as f:
        json.dump({"model": "densenet121", "dataset": "Kaggle Chest X-Ray (Pneumonia)", **metrics}, f, indent=2)
    print(f"  Saved: {out}")
