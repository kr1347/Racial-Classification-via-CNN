# Facial Attribute Classification with Custom CNN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kr1347/Racial-Classification-via-CNN/blob/main/Racial_highest(Final).ipynb)

Multi-class facial attribute classification using a custom CNN architecture with focal loss and class-weighted training to handle severe class imbalance in the UTKFace dataset.

## Architecture

Custom CNN with 4 convolutional blocks:

```
Input(128×128×3)
→ Conv(64) → BN → ReLU → MaxPool
→ Conv(128) → BN → ReLU → MaxPool
→ Conv(256) → BN → ReLU → MaxPool
→ Conv(512) → BN → ReLU → GlobalAvgPool
→ Dense(512) → Dropout → Dense(5, Softmax)
```

## Dataset

**UTKFace** — 20,000+ face images with demographic attribute labels.

- 5 classes (imbalanced distribution)
- Input resolution: 128×128
- Train/Val split: 80/20

## Training Strategy

Two techniques applied to address class imbalance:

**Focal Loss** (Lin et al., 2017) — down-weights easy examples, focuses training on hard minority-class samples:

```python
FL(p_t) = -α_t (1 - p_t)^γ · log(p_t)   # γ=3.0
```

**Class-weighted loss** — manual weight scaling for underrepresented classes (up to 5× for rarest class).

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW |
| Epochs | 15 |
| Batch size | 8 |
| Input size | 128×128 |
| Augmentation | RandomFlip, Brightness, Contrast |

## Usage

Open in Colab. Mount Google Drive and set `zip_file_path` to your UTKFace dataset location. Run all cells.

## References

- Lin, T. et al. (2017). Focal Loss for Dense Object Detection. *ICCV 2017*
- Zhang, Z. et al. (2017). Age progression/regression by conditional adversarial autoencoder. *CVPR 2017* (UTKFace dataset)

