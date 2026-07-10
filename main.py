"""
main.py — Chest X-Ray Pneumonia Detection
------------------------------------------
Entry point for the full pipeline.

Commands:
  # Quick smoke-test (no data needed)
  python main.py

  # Train
  python main.py --mode train

  # Evaluate saved checkpoint on test set
  python main.py --mode evaluate --checkpoint outputs/models/best_model.pth

  # Single image prediction + Grad-CAM
  python main.py --mode predict --image path/to/xray.jpg --checkpoint outputs/models/best_model.pth
"""

import sys
import argparse
import logging
import warnings
from pathlib import Path

import torch
import yaml

warnings.filterwarnings("ignore", category=UserWarning)
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


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"GPU : {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Device: Apple MPS")
    else:
        device = torch.device("cpu")
        logger.info("Device: CPU  (training will be slow — consider a GPU)")
    return device


def ensure_dirs(config: dict):
    for key in ["model_dir", "log_dir", "viz_dir", "reports_dir"]:
        Path(config["output"][key]).mkdir(parents=True, exist_ok=True)


# ─── Modes ────────────────────────────────────────────────────────────────────

def run_demo(config: dict, device: torch.device):
    """Forward-pass smoke-test with random tensors — no real data needed."""
    import torch, numpy as np
    from src.models import build_model
    from src.evaluate import compute_metrics

    logger.info("Running DEMO mode (synthetic data)…")
    model = build_model(
        name=config["model"]["name"],
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
    ).to(device)
    model.eval()

    dummy = torch.randn(4, 3, config["data"]["image_size"], config["data"]["image_size"]).to(device)
    with torch.no_grad():
        out = model(dummy)
    logger.info(f"Forward pass OK  input={list(dummy.shape)}  output={list(out.shape)}")

    # Random baseline metrics
    np.random.seed(42)
    n = 100
    labels = np.random.randint(0, 2, n).tolist()
    preds  = np.random.randint(0, 2, n).tolist()
    probs  = np.random.uniform(0, 1, n).tolist()
    m = compute_metrics(labels, preds, probs)
    print("\n  Demo metrics (random baseline — not meaningful):")
    for k, v in m.items():
        print(f"    {k}: {v:.4f}")
    print("\n  Pipeline OK. To train for real:")
    print("    1. Download dataset (see README → Quick Start)")
    print(f"    2. python main.py --mode train\n")


def run_train(config: dict, device: torch.device):
    from src.dataset import get_dataloaders
    from src.models  import build_model
    from src.train   import train
    from src.evaluate import evaluate

    logger.info("Loading data…")
    loaders = get_dataloaders(
        root_dir=config["data"]["root_dir"],
        image_size=config["data"]["image_size"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    logger.info("Building model…")
    model = build_model(
        name=config["model"]["name"],
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
    ).to(device)

    logger.info("Starting training…")
    model = train(model, loaders, config, device)

    logger.info("Evaluating on test set…")
    evaluate(
        model, loaders["test"], device,
        output_dir=config["output"]["viz_dir"],
        reports_dir=config["output"]["reports_dir"],
    )


def run_evaluate(config: dict, device: torch.device, checkpoint: str):
    from src.dataset  import get_dataloaders
    from src.models   import build_model, load_checkpoint
    from src.evaluate import evaluate

    loaders = get_dataloaders(
        root_dir=config["data"]["root_dir"],
        image_size=config["data"]["image_size"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )
    model = build_model(name=config["model"]["name"], pretrained=False).to(device)
    load_checkpoint(model, checkpoint, device)
    evaluate(
        model, loaders["test"], device,
        output_dir=config["output"]["viz_dir"],
        reports_dir=config["output"]["reports_dir"],
    )


def run_predict(config: dict, device: torch.device, checkpoint: str, image_path: str):
    from src.models  import build_model, load_checkpoint
    from src.predict import predict

    model = build_model(name=config["model"]["name"], pretrained=False).to(device)
    load_checkpoint(model, checkpoint, device)
    predict(
        model, image_path, device,
        image_size=config["data"]["image_size"],
        threshold=config["evaluation"]["threshold"],
        model_name=config["model"]["name"],
        output_dir=config["output"]["viz_dir"],
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Chest X-Ray Pneumonia Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode",       choices=["demo", "train", "evaluate", "predict"],
                   default="demo")
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--checkpoint", default=None, help="Path to best_model.pth")
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

    # CLI overrides
    if args.model:      config["model"]["name"]               = args.model
    if args.epochs:     config["training"]["epochs"]          = args.epochs
    if args.batch_size: config["training"]["batch_size"]      = args.batch_size
    if args.lr:         config["training"]["learning_rate"]   = args.lr

    device = get_device()
    ensure_dirs(config)

    if args.mode == "demo":
        run_demo(config, device)

    elif args.mode == "train":
        run_train(config, device)

    elif args.mode == "evaluate":
        if not args.checkpoint:
            logger.error("--checkpoint is required for evaluate mode")
            sys.exit(1)
        run_evaluate(config, device, args.checkpoint)

    elif args.mode == "predict":
        if not args.checkpoint or not args.image:
            logger.error("--checkpoint and --image are both required for predict mode")
            sys.exit(1)
        run_predict(config, device, args.checkpoint, args.image)


if __name__ == "__main__":
    main()
