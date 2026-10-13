"""
main.py: Chest X-Ray Pneumonia Detection (TensorFlow / Keras)
---------------------------------------------------------------
Entry point for the full pipeline.

Commands:
  # Quick smoke-test (no data needed)
  python main.py

  # Train (then evaluates on the test set and writes Grad-CAM examples)
  python main.py --mode train

  # Evaluate a saved model on the test set
  python main.py --mode evaluate --checkpoint outputs/models/best_model.keras

  # Single image prediction + Grad-CAM
  python main.py --mode predict --image path/to/xray.jpg --checkpoint outputs/models/best_model.keras
"""

import os
import sys
import argparse
import logging
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import keras
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_runtime(config: dict) -> bool:
    """Seeds everything and turns on mixed precision when a GPU is present."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:  # must happen before the GPU is first used
        tf.config.experimental.set_memory_growth(gpu, True)
    keras.utils.set_random_seed(config.get("seed", 42))
    if gpus:
        logger.info(f"GPU : {gpus[0].name}")
        if config["training"].get("mixed_precision", True):
            keras.mixed_precision.set_global_policy("mixed_float16")
            logger.info("Mixed precision: mixed_float16")
    else:
        logger.info("Device: CPU  (training will be slow, consider a GPU such as Colab)")
    return bool(gpus)


def ensure_dirs(config: dict):
    for key in ["model_dir", "log_dir", "viz_dir", "reports_dir"]:
        Path(config["output"][key]).mkdir(parents=True, exist_ok=True)


def load_data(config: dict):
    from src.dataset import get_datasets
    from src.models import get_preprocess

    return get_datasets(
        root_dir=config["data"]["root_dir"],
        image_size=config["data"]["image_size"],
        batch_size=config["training"]["batch_size"],
        preprocess=get_preprocess(config["model"]["name"]),
        seed=config.get("seed", 42),
    )


def evaluate_and_explain(model, data: dict, config: dict, run_info: dict = None):
    """Test-set metrics + plots, then Grad-CAM on real test images."""
    from src.evaluate import evaluate
    from src.gradcam import save_gradcam_examples
    from src.models import get_preprocess, gradcam_layer

    name = config["model"]["name"]
    threshold = config["evaluation"]["threshold"]
    result = evaluate(
        model, data["test"], threshold=threshold,
        output_dir=config["output"]["viz_dir"],
        reports_dir=config["output"]["reports_dir"],
        run_info=run_info,
    )
    save_gradcam_examples(
        model, gradcam_layer(name), get_preprocess(name),
        data["test"]["paths"], data["test"]["labels"], result["probs"],
        output_dir=config["output"]["viz_dir"],
        image_size=config["data"]["image_size"],
        threshold=threshold,
    )
    return result


# ─── Modes ────────────────────────────────────────────────────────────────────

def run_demo(config: dict):
    """Forward pass on random images, no real data needed."""
    from src.models import build_model, get_preprocess
    from src.evaluate import compute_metrics

    logger.info("Running DEMO mode (synthetic data)…")
    size = config["data"]["image_size"]
    model = build_model(
        name=config["model"]["name"],
        image_size=size,
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
    )
    dummy = tf.random.uniform((4, size, size, 3), 0, 255)
    out = model(get_preprocess(config["model"]["name"])(dummy), training=False)
    logger.info(f"Forward pass OK  input={list(dummy.shape)}  output={list(out.shape)}")

    rng = np.random.default_rng(42)
    labels, preds, probs = rng.integers(0, 2, 100), rng.integers(0, 2, 100), rng.uniform(0, 1, 100)
    m = compute_metrics(labels, preds, probs)
    print("\n  Demo metrics (random baseline, not meaningful):")
    for k, v in m.items():
        print(f"    {k}: {v:.4f}")
    print("\n  Pipeline OK. To train for real:")
    print("    1. Download dataset (see README → Setup)")
    print("    2. python main.py --mode train\n")


def run_train(config: dict, on_gpu: bool):
    from src.models import build_model
    from src.train import train

    logger.info("Loading data…")
    data = load_data(config)

    logger.info("Building model…")
    model = build_model(
        name=config["model"]["name"],
        image_size=config["data"]["image_size"],
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
    )

    logger.info("Starting training…")
    model, summary = train(model, data, config)

    run_info = {
        "best_epoch": summary["best_epoch"],
        "epochs_trained": summary["epochs_trained"],
        "best_val_auc": summary["best_val_auc"],
        "device": "GPU" if on_gpu else "CPU",
        "training_config": {**config["training"], "image_size": config["data"]["image_size"],
                            "seed": config.get("seed", 42)},
    }
    logger.info("Evaluating on test set…")
    evaluate_and_explain(model, data, config, run_info)


def run_evaluate(config: dict, checkpoint: str):
    from src.models import load_model

    data = load_data(config)
    model = load_model(checkpoint)
    evaluate_and_explain(model, data, config)


def run_predict(config: dict, checkpoint: str, image_path: str):
    from src.models import load_model, get_preprocess, gradcam_layer
    from src.predict import predict

    name = config["model"]["name"]
    model = load_model(checkpoint)
    predict(
        model, image_path, get_preprocess(name),
        image_size=config["data"]["image_size"],
        threshold=config["evaluation"]["threshold"],
        gradcam_layer=gradcam_layer(name),
        output_dir=config["output"]["viz_dir"],
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Chest X-Ray Pneumonia Detection (TensorFlow / Keras)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode",       choices=["demo", "train", "evaluate", "predict"], default="demo")
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--checkpoint", default=None, help="Path to best_model.keras")
    p.add_argument("--image",      default=None, help="Image path for predict mode")
    # Optional overrides
    p.add_argument("--model",      choices=["densenet121", "resnet50"], default=None)
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config)

    if args.model:      config["model"]["name"]             = args.model
    if args.epochs:     config["training"]["epochs"]        = args.epochs
    if args.batch_size: config["training"]["batch_size"]    = args.batch_size
    if args.lr:         config["training"]["learning_rate"] = args.lr

    on_gpu = setup_runtime(config)
    ensure_dirs(config)

    if args.mode == "demo":
        run_demo(config)

    elif args.mode == "train":
        run_train(config, on_gpu)

    elif args.mode == "evaluate":
        if not args.checkpoint:
            logger.error("--checkpoint is required for evaluate mode")
            sys.exit(1)
        run_evaluate(config, args.checkpoint)

    elif args.mode == "predict":
        if not args.checkpoint or not args.image:
            logger.error("--checkpoint and --image are both required for predict mode")
            sys.exit(1)
        run_predict(config, args.checkpoint, args.image)


if __name__ == "__main__":
    main()
