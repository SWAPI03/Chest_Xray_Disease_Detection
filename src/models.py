"""
models.py
---------
CNN models for chest X-ray pneumonia detection.

Supported backbones:
  - DenseNet121  (primary — mirrors CheXNet, Rajpurkar et al. 2017)
  - ResNet50     (alternative — faster, larger parameter count)

Both use ImageNet pretrained weights and a small custom classification head.
"""

import torch
import torch.nn as nn
from torchvision import models


# ─── DenseNet121 ─────────────────────────────────────────────────────────────

class DenseNet121(nn.Module):
    """
    DenseNet121 backbone + lightweight classification head.

    Architecture mirrors CheXNet (Rajpurkar et al. 2017):
        DenseNet121 features → ReLU → GlobalAvgPool → Dropout → FC(1024→2)
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.densenet121(weights=weights)

        self.features   = base.features          # output: (B, 1024, H, W)
        self.relu       = nn.ReLU(inplace=True)
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ─── ResNet50 ────────────────────────────────────────────────────────────────

class ResNet50(nn.Module):
    """
    ResNet50 backbone + classification head.
    Slightly higher param count than DenseNet121, good accuracy baseline.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = models.resnet50(weights=weights)

        # Keep everything except the original FC
        self.backbone   = nn.Sequential(*list(base.children())[:-1])  # → (B, 2048, 1, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ─── Factory ─────────────────────────────────────────────────────────────────

def build_model(
    name: str = "densenet121",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.5,
) -> nn.Module:
    """
    Build a model by name.

    Args:
        name:        'densenet121' or 'resnet50'
        num_classes: Output neurons (2 for binary classification)
        pretrained:  Load ImageNet weights
        dropout:     Dropout before the final FC layer

    Returns:
        Configured nn.Module
    """
    name = name.lower()

    if name == "densenet121":
        model = DenseNet121(num_classes, pretrained, dropout)
    elif name == "resnet50":
        model = ResNet50(num_classes, pretrained, dropout)
    else:
        raise ValueError(f"Unknown model '{name}'. Choose 'densenet121' or 'resnet50'.")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model : {name.upper()}  |  Params: {total/1e6:.1f}M  |  Trainable: {trainable/1e6:.1f}M")
    return model


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(model: nn.Module, optimizer, epoch: int, metrics: dict, path: str):
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics":              metrics,
    }, path)


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded checkpoint: {path}  (epoch {ckpt.get('epoch', '?')})")
    return ckpt
