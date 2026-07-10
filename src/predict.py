"""
predict.py
----------
Single-image inference with optional Grad-CAM visualization.

Usage:
    python main.py --mode predict --image path/to/xray.jpg --checkpoint outputs/models/best_model.pth
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.dataset import get_val_transforms, CLASSES
from src.gradcam import GradCAM


def predict(
    model: nn.Module,
    image_path: str,
    device: torch.device,
    image_size: int = 224,
    threshold: float = 0.5,
    save_gradcam: bool = True,
    model_name: str = "densenet121",
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Run inference on a single chest X-ray image.

    Returns:
        {
          'prediction':   'Normal' | 'Pneumonia',
          'confidence':   0.98,
          'prob_normal':  0.02,
          'prob_pneumonia': 0.98,
        }
    """
    # ── Load & preprocess ─────────────────────────────────────────────────────
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = get_val_transforms(image_size)
    tensor    = transform(image=image)["image"].unsqueeze(0).to(device)

    # ── Forward pass ──────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
    probs   = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx = int(np.argmax(probs))

    result = {
        "prediction":     CLASSES[pred_idx],
        "confidence":     float(probs[pred_idx]),
        "prob_normal":    float(probs[0]),
        "prob_pneumonia": float(probs[1]),
    }

    # ── Print result ──────────────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print(f"  Image      : {Path(image_path).name}")
    print(f"  Prediction : {result['prediction']}")
    print(f"  Confidence : {result['confidence']:.1%}")
    print(f"  Normal     : {result['prob_normal']:.1%}")
    print(f"  Pneumonia  : {result['prob_pneumonia']:.1%}")
    print(f"{'─'*40}\n")

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    if save_gradcam and output_dir:
        try:
            cam = GradCAM(model, model_name=model_name)
            with torch.enable_grad():
                heatmap, overlay = cam.generate(tensor.clone(), target_class=pred_idx)

            stem    = Path(image_path).stem
            viz_dir = Path(output_dir)
            viz_dir.mkdir(parents=True, exist_ok=True)

            GradCAM.save_comparison(
                tensor.cpu(), heatmap, overlay,
                prediction=result["prediction"],
                confidence=result["confidence"],
                output_path=str(viz_dir / f"{stem}_gradcam.png"),
            )
            cam.remove_hooks()
        except Exception as e:
            print(f"  Grad-CAM warning: {e}")

    return result
