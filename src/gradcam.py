"""
gradcam.py
----------
Grad-CAM: highlights the chest X-ray regions that most influenced the prediction.

Supports DenseNet121 (hooks on denseblock4) and ResNet50 (hooks on layer4).
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: nn.Module, model_name: str = "densenet121"):
        self.model      = model
        self.model_name = model_name.lower()
        self.gradients  = None
        self.activations = None
        self._hooks     = []
        self._register_hooks()

    def _target_layer(self) -> nn.Module:
        if self.model_name == "densenet121":
            return self.model.features.denseblock4
        if self.model_name == "resnet50":
            return list(self.model.backbone.children())[-2]
        raise ValueError(f"Unsupported model: {self.model_name}")

    def _register_hooks(self):
        layer = self._target_layer()
        self._hooks.append(layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach())
        ))
        self._hooks.append(layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach())
        ))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (heatmap, overlay) for the given image tensor (1, C, H, W).
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device)

        # Forward
        output = self.model(image_tensor)
        probs  = torch.softmax(output, dim=1)
        if target_class is None:
            target_class = probs.argmax(dim=1).item()

        # Backward
        self.model.zero_grad()
        probs[0, target_class].backward()

        # Grad-CAM
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam     = cam.squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        h, w    = image_tensor.shape[2], image_tensor.shape[3]
        heatmap = cv2.resize(cam, (w, h))
        overlay = self._overlay(image_tensor, heatmap)
        return heatmap, overlay

    def _overlay(self, tensor: torch.Tensor, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img  = tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        img  = (img * std + mean).clip(0, 1)
        img  = (img * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        heat_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(img_bgr, 1 - alpha, heat_color, alpha, 0)

    @staticmethod
    def save_comparison(
        image_tensor: torch.Tensor,
        heatmap: np.ndarray,
        overlay: np.ndarray,
        prediction: str,
        confidence: float,
        output_path: str,
    ):
        """Save original | heatmap | overlay side-by-side."""
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img  = image_tensor.squeeze().numpy().transpose(1, 2, 0)
        img  = (img * std + mean).clip(0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle(
            f"Grad-CAM  —  Prediction: {prediction}  |  Confidence: {confidence:.1%}",
            fontsize=13, fontweight="bold",
        )

        axes[0].imshow(img, cmap="gray" if img.std() < 0.08 else None)
        axes[0].set_title("Original X-Ray", fontweight="bold")
        axes[0].axis("off")

        im = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("Activation Heatmap", fontweight="bold")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Grad-CAM Overlay", fontweight="bold")
        axes[2].axis("off")

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved Grad-CAM: {output_path}")
