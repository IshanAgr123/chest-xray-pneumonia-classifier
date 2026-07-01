# Chest X-Ray Pneumonia Classifier

A deep-learning image classifier that detects pneumonia from chest X-ray images, built by fine-tuning a pretrained ResNet18 with PyTorch. The project covers the full computer-vision workflow: data preparation, class-imbalance handling, transfer learning, evaluation with clinically relevant metrics, and model interpretability with Grad-CAM.

## Overview

Pneumonia detection from chest X-rays is an image classification problem where raw accuracy is misleading, because the dataset is imbalanced and the cost of a missed diagnosis (false negative) is far higher than a false alarm. This project therefore optimises for **recall on the pneumonia class** while reporting threshold-independent metrics, and uses Grad-CAM to verify the model bases its decisions on lung pathology rather than image artifacts.

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — ~5,800 labelled pediatric chest X-rays (Kermany et al., Guangzhou Women's and Children's Medical Center), split into `train`, `test`, and `val`.

- Training distribution: 1,341 NORMAL vs 3,875 PNEUMONIA (**~2.9 : 1 imbalance**)
- The official validation set has only 16 images, so a 10% validation split is carved from the training data instead.

> The dataset is **not** included in this repo. Download it from the Kaggle link above.

## Approach

- **Transfer learning:** ResNet18 pretrained on ImageNet, with its final layer replaced for 2-class output and the network fine-tuned on the X-rays.
- **Imbalance handling:** `WeightedRandomSampler` (resampling) plus data augmentation (random flip, rotation, brightness/contrast jitter).
- **Preprocessing:** resize to 224×224, ImageNet normalisation.
- **Training:** Adam (lr 1e-4), batch size 32, 8 epochs, cross-entropy loss; best checkpoint selected on validation accuracy.

## Results (held-out test set, 624 images)

| Metric | Score |
|---|---|
| ROC-AUC | **0.96** |
| Pneumonia recall | **0.99** |
| Accuracy | 86% |
| Pneumonia F1 | 0.90 |
| Normal F1 | 0.78 |

**Confusion matrix** (positive = pneumonia): 386 true positives, 4 false negatives, 153 true negatives, 81 false positives. The model is deliberately cautious — it missed only 4 of 390 pneumonia cases, trading some false alarms on healthy patients for very high sensitivity, which is the appropriate trade-off for a medical screening tool.

The high ROC-AUC (0.96) alongside the lower raw accuracy reflects a decision-threshold effect rather than weak discrimination; tuning the threshold recovers normal-class precision.

## Interpretability

Grad-CAM heatmaps over the final convolutional block confirm that pneumonia predictions are driven by the lung fields rather than text labels or image borders.

## How to run

The project runs on Google Colab or Kaggle Notebooks with a free GPU.

```bash
pip install -r requirements.txt
```

1. Download the dataset from Kaggle and point `DATA_DIR` to its `chest_xray` folder.
2. Open the notebook (or `pneumonia_resnet18.py`) and run the cells top to bottom.
3. Training takes roughly 10–20 minutes on a T4 GPU.

## Limitations

- Trained on a single-source, pediatric dataset — may not generalise to adults, other equipment, or other populations (distribution shift).
- Binary only; does not distinguish bacterial vs viral pneumonia or other lung conditions.
- The decision threshold is not tuned, which lowers accuracy on normal cases.
- Not a validated medical device; for educational use only.

## Future work

- Decision-threshold tuning and probability calibration
- k-fold cross-validation and patient-level splits to rule out leakage
- Benchmarking against DenseNet121 / EfficientNet

## Acknowledgements

Dataset: Kermany et al., *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning* (2018). Built with PyTorch and torchvision.
