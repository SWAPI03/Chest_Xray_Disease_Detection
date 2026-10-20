# Evaluation Report

**Model:** densenet121 (ImageNet pretrained, fine-tuned with TensorFlow/Keras)
**Framework:** TensorFlow 2.21.0 / Keras 3.13.2
**Test set:** 624 images (234 Normal, 390 Pneumonia)
**Decision threshold:** 0.5

## Summary metrics

| Metric | Value |
|---|---|
| Accuracy | 0.8462 |
| AUC-ROC | 0.9429 |
| Recall (sensitivity) | 0.9974 |
| Specificity | 0.5940 |
| Precision | 0.8037 |
| F1 score | 0.8902 |

## Confusion matrix

| | Predicted Normal | Predicted Pneumonia |
|---|---|---|
| **Actual Normal** | 139 | 95 |
| **Actual Pneumonia** | 1 | 389 |

## Classification report

```
              precision    recall  f1-score   support

      Normal     0.9929    0.5940    0.7433       234
   Pneumonia     0.8037    0.9974    0.8902       390

    accuracy                         0.8462       624
   macro avg     0.8983    0.7957    0.8167       624
weighted avg     0.8746    0.8462    0.8351       624

```

## Training run

```json
{
  "best_epoch": 14,
  "epochs_trained": 16,
  "best_val_auc": 0.999580979347229,
  "device": "GPU",
  "training_config": {
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "scheduler": "cosine",
    "patience": 5,
    "use_class_weights": true,
    "mixed_precision": true,
    "image_size": 224,
    "seed": 42
  }
}
```
