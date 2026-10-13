"""
predict.py
----------
Single-image inference with optional Grad-CAM visualization.

Usage:
    python main.py --mode predict --image path/to/xray.jpg --checkpoint outputs/models/best_model.keras
"""

from pathlib import Path
from typing import Callable, Dict, Optional

import tensorflow as tf
import keras

from src.dataset import CLASSES, load_image
from src.gradcam import GradCAM, save_comparison


def predict(
    model: keras.Model,
    image_path: str,
    preprocess: Callable,
    image_size: int = 224,
    threshold: float = 0.5,
    gradcam_layer: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Run inference on a single chest X-ray image.

    Returns:
        {
          'prediction':     'Normal' | 'Pneumonia',
          'confidence':     0.98,
          'prob_normal':    0.02,
          'prob_pneumonia': 0.98,
        }
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    image = load_image(image_path, image_size)
    x = preprocess(tf.convert_to_tensor(image[None], tf.float32))
    p = float(model.predict(x, verbose=0)[0, 0])
    pred_idx = int(p >= threshold)

    result = {
        "prediction":     CLASSES[pred_idx],
        "confidence":     p if pred_idx == 1 else 1 - p,
        "prob_normal":    1 - p,
        "prob_pneumonia": p,
    }

    print(f"\n{'─'*40}")
    print(f"  Image      : {Path(image_path).name}")
    print(f"  Prediction : {result['prediction']}")
    print(f"  Confidence : {result['confidence']:.1%}")
    print(f"  Normal     : {result['prob_normal']:.1%}")
    print(f"  Pneumonia  : {result['prob_pneumonia']:.1%}")
    print(f"{'─'*40}\n")

    if gradcam_layer and output_dir:
        cam = GradCAM(model, gradcam_layer, preprocess)
        heatmap, over, _ = cam.generate(image, target_class=pred_idx)
        save_comparison(
            image, heatmap, over,
            prediction=result["prediction"],
            confidence=result["confidence"],
            output_path=str(Path(output_dir) / f"{Path(image_path).stem}_gradcam.png"),
        )

    return result
