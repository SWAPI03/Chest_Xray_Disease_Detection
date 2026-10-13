"""
evaluate.py
-----------
Evaluation metrics and visualizations for pneumonia detection.

Metrics: Accuracy, Precision, Recall, Specificity, F1, AUC-ROC
Plots:   Confusion Matrix, ROC Curve, prediction examples
Saves:   outputs/reports/metrics.json and outputs/reports/evaluation_report.md
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    classification_report,
)

from src.dataset import CLASSES, load_image


# ─── Core metrics ─────────────────────────────────────────────────────────────

def compute_metrics(labels, preds, probs) -> Dict[str, float]:
    """Return accuracy / precision / recall / specificity / f1 / auc."""
    labels, preds, probs = np.asarray(labels), np.asarray(preds), np.asarray(probs)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "accuracy":    float(accuracy_score(labels, preds)),
        "precision":   float(precision_score(labels, preds, zero_division=0)),
        "recall":      float(recall_score(labels, preds, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1":          float(f1_score(labels, preds, zero_division=0)),
        "auc":         float(roc_auc_score(labels, probs)),
    }


def predict_probs(model: keras.Model, ds: tf.data.Dataset) -> np.ndarray:
    """P(pneumonia) for every image in a (non-shuffled) dataset."""
    return model.predict(ds, verbose=0).ravel().astype(float)


# ─── Full test-set evaluation ────────────────────────────────────────────────

def evaluate(
    model: keras.Model,
    split: dict,
    threshold: float = 0.5,
    output_dir: Optional[str] = None,
    reports_dir: Optional[str] = None,
    run_info: Optional[dict] = None,
) -> Dict:
    """
    Run the model on a split ({'ds', 'paths', 'labels'}), print a report,
    save plots, metrics.json and evaluation_report.md.

    Returns {'metrics', 'probs', 'preds'}.
    """
    labels = split["labels"]
    probs  = predict_probs(model, split["ds"])
    preds  = (probs >= threshold).astype(int)
    metrics = compute_metrics(labels, preds, probs)
    report  = classification_report(labels, preds, target_names=CLASSES, digits=4)
    _print_report(report, metrics)

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        _plot_confusion_matrix(labels, preds, output_dir, model.name)
        _plot_roc_curve(labels, probs, metrics["auc"], threshold, output_dir, model.name)
        plot_prediction_examples(split["paths"], labels, probs, threshold,
                                 Path(output_dir) / "prediction_examples.png")

    if reports_dir:
        _save_reports(labels, preds, metrics, report, threshold, model.name, reports_dir, run_info or {})

    return {"metrics": metrics, "probs": probs, "preds": preds}


# ─── Console report ───────────────────────────────────────────────────────────

def _print_report(report: str, metrics: dict):
    print("\n" + "=" * 50)
    print("  TEST SET RESULTS")
    print("=" * 50)
    print(report)
    for k in ["auc", "accuracy", "precision", "recall", "specificity", "f1"]:
        print(f"  {k.upper():<12}: {metrics[k]:.4f}")
    print("=" * 50 + "\n")


# ─── Confusion matrix ─────────────────────────────────────────────────────────

def _plot_confusion_matrix(labels, preds, output_dir: str, model_name: str):
    cm      = confusion_matrix(labels, preds, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Confusion Matrix: {model_name}  (Test Set, n={len(labels)})",
                 fontsize=12, fontweight="bold")

    for ax, data, title, fmt, cmap in zip(
        axes, [cm, cm_norm], ["Counts", "Normalised"], ["d", ".2f"], ["Blues", "Greens"],
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

def _plot_roc_curve(labels, probs, auc: float, threshold: float, output_dir: str, model_name: str):
    fpr, tpr, thresholds = roc_curve(labels, probs)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2.5, label=f"{model_name}  (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.10, color="#2196F3")

    idx = np.argmin(np.abs(thresholds - threshold))
    ax.scatter(fpr[idx], tpr[idx], color="#F44336", s=100, zorder=5,
               label=f"Threshold {threshold}  (FPR={fpr[idx]:.2f}, TPR={tpr[idx]:.2f})")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.03])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title(f"ROC Curve: Pneumonia Detection\n(Test Set n={len(labels)})",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = Path(output_dir) / "roc_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─── Prediction examples ──────────────────────────────────────────────────────

def plot_prediction_examples(
    paths: List[str], labels, probs, threshold: float, out_path: Path,
    n_correct: int = 6, n_wrong: int = 2, seed: int = 42,
):
    """Grid of real test images: some correct predictions and some mistakes."""
    labels, probs = np.asarray(labels), np.asarray(probs)
    preds = (probs >= threshold).astype(int)
    rng = np.random.default_rng(seed)
    correct = np.flatnonzero(preds == labels)
    wrong   = np.flatnonzero(preds != labels)
    picks = list(rng.choice(correct, min(n_correct, len(correct)), replace=False))
    picks += list(rng.choice(wrong, min(n_wrong, len(wrong)), replace=False))

    cols = 4
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.4 * rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, i in zip(np.ravel(axes), picks):
        conf = probs[i] if preds[i] == 1 else 1 - probs[i]
        ok = preds[i] == labels[i]
        ax.imshow(load_image(paths[i]))
        ax.set_title(f"True: {CLASSES[labels[i]]}\nPred: {CLASSES[preds[i]]} ({conf:.1%})",
                     color="#2E7D32" if ok else "#C62828", fontsize=11, fontweight="bold")
    fig.suptitle("Test-set predictions (green = correct, red = wrong)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ─── Saved reports ────────────────────────────────────────────────────────────

def _save_reports(labels, preds, metrics, report, threshold, model_name, reports_dir, run_info):
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    per_class = classification_report(labels, preds, target_names=CLASSES, output_dict=True)
    payload = {
        "model": model_name,
        "framework": f"TensorFlow {tf.__version__} / Keras {keras.__version__}",
        "dataset": "Kaggle Chest X-Ray (Pneumonia)",
        "test_set_size": int(len(labels)),
        "threshold": threshold,
        "metrics": metrics,
        "per_class": {c: {k: float(v) for k, v in per_class[c].items()} for c in CLASSES},
        "confusion_matrix": {
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        },
        **run_info,
    }
    out = Path(reports_dir) / "metrics.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved: {out}")

    m = metrics
    md = f"""# Evaluation Report

**Model:** {model_name} (ImageNet pretrained, fine-tuned with TensorFlow/Keras)
**Framework:** {payload['framework']}
**Test set:** {len(labels)} images ({int((np.asarray(labels) == 0).sum())} Normal, {int((np.asarray(labels) == 1).sum())} Pneumonia)
**Decision threshold:** {threshold}

## Summary metrics

| Metric | Value |
|---|---|
| Accuracy | {m['accuracy']:.4f} |
| AUC-ROC | {m['auc']:.4f} |
| Recall (sensitivity) | {m['recall']:.4f} |
| Specificity | {m['specificity']:.4f} |
| Precision | {m['precision']:.4f} |
| F1 score | {m['f1']:.4f} |

## Confusion matrix

| | Predicted Normal | Predicted Pneumonia |
|---|---|---|
| **Actual Normal** | {tn} | {fp} |
| **Actual Pneumonia** | {fn} | {tp} |

## Classification report

```
{report}
```
"""
    if run_info:
        md += "\n## Training run\n\n```json\n" + json.dumps(run_info, indent=2) + "\n```\n"
    out_md = Path(reports_dir) / "evaluation_report.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"  Saved: {out_md}")
