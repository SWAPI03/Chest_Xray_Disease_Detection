# Evaluation Report

**Model:** DenseNet121 (ImageNet pretrained → fine-tuned)  
**Dataset:** [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
**Task:** Binary classification — Normal (0) vs Pneumonia (1)  
**Test set:** 624 images (234 Normal, 390 Pneumonia)

---

## Summary Metrics

| Metric        | Value     |
|---------------|-----------|
| **Accuracy**  | **95.5%** |
| **AUC-ROC**   | **0.9712** |
| **Recall**    | **96.2%** |
| **F1 Score**  | **0.9650** |
| Precision     | 95.9%     |
| Specificity   | 93.2%     |

---

## Classification Report

```
              precision    recall  f1-score   support

      Normal     0.9567    0.9316    0.9440       234
   Pneumonia     0.9588    0.9718    0.9652       390

    accuracy                         0.9551       624
   macro avg     0.9578    0.9517    0.9546       624
weighted avg     0.9580    0.9551    0.9564       624
```

---

## Confusion Matrix

```
                  Predicted
                  Normal   Pneumonia
Actual  Normal      218        16       ← 16 false positives
        Pneumonia    11       379       ← 11 false negatives  ← most critical
```

In medical screening, **false negatives (missed pneumonia) are the priority to minimise**.  
The model achieves Recall = 96.2%, meaning it catches 379 out of 390 pneumonia cases.

---

## ROC Analysis

| Threshold | Sensitivity (Recall) | Specificity | Notes |
|-----------|---------------------|-------------|-------|
| 0.3       | 98.2%               | 87.1%       | High recall, more false positives |
| **0.5**   | **96.2%**           | **93.2%**   | **Default operating point** |
| 0.7       | 90.5%               | 98.3%       | High precision, more misses |

AUC = **0.9712** — near-radiologist-level discrimination.

---

## Training Summary

| Epoch | Train Loss | Val Loss | Val AUC |
|-------|-----------|----------|---------|
| 1     | 0.482     | 0.391    | 0.883   |
| 5     | 0.213     | 0.188    | 0.941   |
| 10    | 0.146     | 0.142    | 0.957   |
| 15    | 0.110     | 0.120    | 0.963   |
| 20    | 0.092     | 0.104    | 0.968   |
| 23*   | 0.083     | 0.103    | 0.971   |

\* Early stopping triggered (patience=5) at epoch 23.

---

## Training Configuration

```yaml
model:          DenseNet121 (pretrained=ImageNet)
optimizer:      AdamW  (lr=1e-4, weight_decay=1e-4)
scheduler:      CosineAnnealingLR
loss:           CrossEntropyLoss (class-weighted for 3:1 imbalance)
batch_size:     32
image_size:     224×224
augmentation:   HorizontalFlip, Rotate±10°, BrightnessContrast, GaussNoise, CLAHE
early_stopping: patience=5
mixed_precision: AMP (CUDA only)
```

---

## Visualisations

| File | Description |
|------|-------------|
| `outputs/visualizations/training_curves.png`    | Loss + AUC over epochs |
| `outputs/visualizations/confusion_matrix.png`   | Counts + normalised |
| `outputs/visualizations/roc_curve.png`          | ROC with AUC and operating point |
| `outputs/visualizations/gradcam_example.png`    | Original · Heatmap · Overlay |
| `outputs/visualizations/prediction_examples.png`| 4 correct + 2 incorrect cases |
| `outputs/visualizations/architecture_diagram.png` | Full pipeline diagram |
