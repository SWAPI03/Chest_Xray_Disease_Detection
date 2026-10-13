# Chest X-Ray Pneumonia Detection

Classifying chest X-rays as **Normal or Pneumonia** with a DenseNet121 CNN fine-tuned in **TensorFlow/Keras** on the [Kaggle Chest X-Ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). Grad-CAM shows which part of the X-ray the model is actually looking at.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00)](https://www.tensorflow.org/)
[![Keras 3](https://img.shields.io/badge/Keras-3-D00000)](https://keras.io/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Chest%20X--Ray-20BEFF)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SWAPI03/Chest_Xray_Disease_Detection/blob/main/notebooks/train_colab.ipynb)

---

## Results

Results are being regenerated from a fresh training run of the TensorFlow version. Metrics and figures will show up here once it finishes.

---

## How it works

```
Kaggle dataset  →  tf.data (decode, resize, augment)  →  DenseNet121  →  model.fit  →  evaluate  →  Grad-CAM
```

- **Input pipeline (`src/dataset.py`)**: `tf.data` decodes each JPEG, repeats the gray channel to 3 channels (the ImageNet backbone expects RGB), resizes to 224×224 and caches the decoded images after the first epoch.
- **Augmentation (train only)**: horizontal flip, ±10° rotation, brightness/contrast jitter and light Gaussian noise, all done per image with `tf` ops.
- **Model (`src/models.py`)**: `keras.applications.DenseNet121` with ImageNet weights, then GlobalAveragePooling → Dropout(0.5) → Dense(1, sigmoid). Inputs use the same ImageNet mean/std normalisation the weights were trained with.
- **Class imbalance**: the training set is about 3:1 Pneumonia:Normal, so `model.fit` gets inverse-frequency `class_weight`s.
- **Training (`src/train.py`)**: AdamW (weight decay 1e-4, gradient clipping 1.0), cosine learning-rate decay, mixed precision on GPU. `ModelCheckpoint` and `EarlyStopping` both watch validation AUC.
- **Evaluation (`src/evaluate.py`)**: scikit-learn metrics on the held-out test set, confusion matrix, ROC curve, and a grid of real test predictions.
- **Grad-CAM (`src/gradcam.py`)**: a second Keras model returns the last conv feature map with the prediction, `tf.GradientTape` gives the gradients, and the pooled gradients weight the feature map into a heatmap.

---

## Dataset

[Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), originally from Guangzhou Women and Children's Medical Center.

The Kaggle `val/` folder only has 16 images, so `data/prepare_data.py` moves 10% of `train/` into `val/` with a fixed seed. The 624-image `test/` folder is never touched during training.

---

## Why DenseNet121

DenseNet121 was used in [CheXNet](https://arxiv.org/abs/1711.05225) (Rajpurkar et al., 2017), which reached radiologist-level performance on chest X-ray classification. The dense connections let gradients flow cleanly to early layers and help the model pick up fine texture differences in lung tissue. ResNet50 is also supported (`--model resnet50`).

---

## Project structure

```
configs/config.yaml          all settings
data/prepare_data.py         checks the dataset and builds the validation split
main.py                      CLI: demo | train | evaluate | predict
src/dataset.py               tf.data pipeline + augmentation
src/models.py                Keras models + matching preprocessing
src/train.py                 compile, fit, callbacks, training curves
src/evaluate.py              metrics, confusion matrix, ROC, prediction grid
src/gradcam.py               Grad-CAM with tf.GradientTape
src/predict.py               single-image inference
notebooks/train_colab.ipynb  end-to-end run on a free Colab GPU
outputs/                     figures, reports, logs (model file is not committed)
```

---

## Run it

**On Colab (GPU, no setup):** open the notebook with the badge above, pick a T4 GPU runtime and run all cells. It downloads the data, trains, evaluates and writes every figure.

**Locally:**
```bash
pip install -r requirements.txt

kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
mv data/chest_xray data/raw
python data/prepare_data.py

python main.py --mode train
```

Evaluate a saved model:
```bash
python main.py --mode evaluate --checkpoint outputs/models/best_model.keras
```

Predict on a single image (also saves its Grad-CAM):
```bash
python main.py --mode predict --checkpoint outputs/models/best_model.keras --image path/to/xray.jpeg
```

Quick smoke test (no data needed):
```bash
python main.py
```

TensorFlow only uses an NVIDIA GPU on Linux or WSL2, so on native Windows it trains on the CPU. That works but is slow, which is why the Colab notebook exists.

---

## Config

Everything is in `configs/config.yaml`:

```yaml
model:
  name: densenet121   # or resnet50

training:
  epochs: 20
  batch_size: 32
  learning_rate: 0.0001
  scheduler: cosine   # cosine | step | plateau
```

---

## Saved model

The trained model (`outputs/models/best_model.keras`) isn't committed because it's too large for git. Retrain with `python main.py --mode train` or the Colab notebook.

---

## References

- CheXNet, Rajpurkar et al. (2017): [arXiv:1711.05225](https://arxiv.org/abs/1711.05225)
- DenseNet, Huang et al. (2017): [arXiv:1608.06993](https://arxiv.org/abs/1608.06993)
- Grad-CAM, Selvaraju et al. (2017): [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
