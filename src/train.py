"""
train.py
--------
Training loop for chest X-ray pneumonia detection.

Features:
  - Mixed-precision training (AMP) for faster GPU training
  - Cosine / Step / Plateau LR scheduling
  - Early stopping on validation AUC
  - TensorBoard logging
  - Best-model checkpoint saving
"""

import time
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.evaluate import compute_metrics

logger = logging.getLogger(__name__)


# ─── Scheduler factory ────────────────────────────────────────────────────────

def _get_scheduler(optimizer, config: dict):
    name = config["training"]["scheduler"]
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["training"]["epochs"], eta_min=1e-6
        )
    if name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    if name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5
        )
    raise ValueError(f"Unknown scheduler: {name}")


# ─── Early stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.counter  = 0
        self.best     = None

    def step(self, score: float) -> bool:
        if self.best is None or score > self.best + 1e-4:
            self.best    = score
            self.counter = 0
            return False
        self.counter += 1
        logger.info(f"  EarlyStopping: {self.counter}/{self.patience}")
        return self.counter >= self.patience


# ─── One epoch ────────────────────────────────────────────────────────────────

def _run_epoch(
    model, loader: DataLoader, criterion, optimizer,
    device, scaler, phase: str,
) -> Dict[str, float]:
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.set_grad_enabled(is_train):
        for batch in tqdm(loader, desc=f"  {phase.capitalize():5s}", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with autocast(enabled=(scaler is not None)):
                logits = model(images)
                loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            total_loss += loss.item() * images.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            all_probs.extend(probs.detach().cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


# ─── Main training function ───────────────────────────────────────────────────

def train(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    config: dict,
    device: torch.device,
) -> nn.Module:
    """
    Train the model and return the best checkpoint.

    Saves best_model.pth to outputs/models/ whenever validation AUC improves.
    Logs to TensorBoard in outputs/logs/.
    """
    cfg_t = config["training"]
    epochs     = cfg_t["epochs"]
    lr         = cfg_t["learning_rate"]
    wd         = cfg_t["weight_decay"]
    use_amp    = cfg_t.get("mixed_precision", True)
    patience   = cfg_t["patience"]

    model_dir = Path(config["output"]["model_dir"])
    log_dir   = Path(config["output"]["log_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Loss — class-weighted to handle 3:1 imbalance
    if cfg_t.get("use_class_weights", True):
        cw = dataloaders["train"].dataset.get_class_weights().to(device)
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = _get_scheduler(optimizer, config)
    scaler    = GradScaler() if use_amp and device.type == "cuda" else None
    writer    = SummaryWriter(log_dir=str(log_dir))
    stopper   = EarlyStopping(patience=patience)

    best_auc  = 0.0
    best_path = model_dir / config["output"]["best_model_name"]

    print(f"\n{'='*55}")
    print(f"  Model   : {config['model']['name'].upper()}")
    print(f"  Epochs  : {epochs}  |  LR: {lr}  |  Device: {device}")
    print(f"  AMP     : {scaler is not None}")
    print(f"{'='*55}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        print(f"Epoch [{epoch}/{epochs}]")

        train_m = _run_epoch(model, dataloaders["train"], criterion, optimizer, device, scaler, "train")
        val_m   = _run_epoch(model, dataloaders["val"],   criterion, None,      device, None,   "val")

        # LR step
        if config["training"]["scheduler"] == "plateau":
            scheduler.step(val_m["auc"])
        else:
            scheduler.step()

        # Log
        for k, v in train_m.items():
            writer.add_scalar(f"Train/{k}", v, epoch)
        for k, v in val_m.items():
            writer.add_scalar(f"Val/{k}", v, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"  Train  Loss: {train_m['loss']:.4f}  AUC: {train_m['auc']:.4f}  Acc: {train_m['accuracy']:.4f}\n"
            f"  Val    Loss: {val_m['loss']:.4f}  AUC: {val_m['auc']:.4f}  Acc: {val_m['accuracy']:.4f}"
            f"  ({time.time()-t0:.0f}s)"
        )

        # Save best
        if val_m["auc"] > best_auc:
            best_auc = val_m["auc"]
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics":              val_m,
                "config":               config,
            }, best_path)
            print(f"  ✓ Best model saved  (Val AUC: {best_auc:.4f})")

        if stopper.step(val_m["auc"]):
            print(f"\n  Early stopping at epoch {epoch}.")
            break

    writer.close()
    print(f"\nTraining done.  Best Val AUC: {best_auc:.4f}  →  {best_path}")

    # Return best model
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model
