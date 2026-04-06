# DINOv2 Offroad Semantic Segmentation Ensemble

**Duality AI Hackathon Submission** — Ensemble of 3 complementary models for desert terrain segmentation.

---

## 📋 Quick Reference

| Metric | Value |
|--------|-------|
| Backbone | DINOv2 vits14 (384-dim) |
| Image Size | 252×462 (18×33 token grid) |
| Classes | 10 (Background, Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Logs, Rocks, Landscape, Sky) |
| Decoder | SegmentationHeadConvNeXt (5.19M params) |
| Ensemble | 3 models with weighted soft-voting |
| TTA | 2-variant (original + horizontal flip) |

---

## 🚀 Installation

```bash
# Clone or extract submission
cd submission/

# Install dependencies
pip install -r requirements.txt

# Download DINOv2 (automatic on first run)
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+ (with CUDA for GPU inference)
- 8GB+ RAM (16GB recommended)
- GPU optional but recommended (CPU inference ~30-60s per image)

---

## 📁 Dataset Structure

Expected directory layout:

```
data/
├── Offroad_Segmentation_Training_Dataset/
│   ├── train/
│   │   ├── Color_Images/          # 2,857 training images (960×540 PNG)
│   │   └── Segmentation/          # 16-bit masks
│   └── val/
│       ├── Color_Images/          # 317 validation images
│       └── Segmentation/          # 16-bit masks
└── Offroad_Segmentation_testImages/
    └── Color_Images/              # 1,002 test images
```

---

## 🎯 Usage

### 1. Validation Evaluation (with metrics)

```bash
# Single model, no TTA
python tta.py --model_path model_augmented_best.pth --mode val --data_dir data/val

# Single model with TTA
python tta.py --model_path model_augmented_best.pth --mode val --data_dir data/val --tta
```

### 2. Test Inference (generate predictions)

```bash
# Without TTA
python tta.py --model_path model_augmented_best.pth --mode test \
  --data_dir data/Offroad_Segmentation_testImages \
  --output_dir predictions

# With TTA (recommended)
python tta.py --model_path model_augmented_best.pth --mode test \
  --data_dir data/Offroad_Segmentation_testImages \
  --output_dir predictions --tta
```

Output structure:
```
predictions/
├── masks/           # Raw class ID masks (uint8 PNG)
└── masks_color/     # Colorized RGB masks
```

### 3. Ensemble Inference (3 models)

The ensemble script (`ensemble_all3.py`) is designed for Kaggle but can run locally:

```python
# Edit CONFIG paths in ensemble_all3.py, then:
python ensemble_all3.py
```

This produces:
- Validation mIoU for each member + ensemble
- Test predictions for all 1,002 images
- Per-class IoU bar charts
- Side-by-side comparison panels

### 4. Visualization

```bash
# Confusion matrix
python visualize.py --mode confusion --pred_dir predictions/masks --gt_dir data/val/Segmentation

# Per-class IoU chart
python visualize.py --mode per_class_iou --metrics_file evaluation_metrics.txt

# Failure case gallery (lowest IoU images)
python visualize.py --mode failures --pred_dir predictions/masks \
  --gt_dir data/val/Segmentation --img_dir data/val/Color_Images --n 5
```

---

## 🏗️ Model Architecture

### Backbone: DINOv2 vits14
- Pre-trained on ImageNet-1K via self-supervised learning
- Frozen during inference (eval mode)
- Output: 384-dimensional patch tokens (18×33 grid for 252×462 input)

### Decoder: SegmentationHeadConvNeXt
```
Input: [B, 594, 384] (594 = 18×33 patch tokens)
  ↓
Reshape → [B, 384, 18, 33]
  ↓
Stem Conv (7×7) → 256 channels + GELU
  ↓
Block1: Depthwise Conv (7×7, groups=256) → PW Conv → GELU
  ↓
Block2: Conv (3×3) → 128 channels + GELU
  ↓
Dropout2d(0.1)
  ↓
Classifier Conv (1×1) → 10 classes
  ↓
Upsample bilinear → 252×462
```

**Total parameters:** 5,192,074

---

## 🧪 Three-Model Ensemble

| Member | Strategy | Val mIoU | Weight |
|--------|----------|----------|--------|
| Member 1 | Fine-tuning (unfreeze last 4 blocks) | 0.4974 | 0.40 |
| Member 2 | Augmentation + class weights | ~0.47 | 0.30 |
| Member 3 | OneCycleLR hyperparameter search | 0.4674 | 0.30 |
| **Ensemble** | Weighted soft-vote + TTA | **TBD** | 1.00 |

### Ensemble Strategy
1. Load shared DINOv2 backbone (frozen)
2. Load 3 independent decoder heads
3. For each input, get softmax probabilities from each head
4. Weighted average: `0.4×M1 + 0.3×M2 + 0.3×M3`
5. Apply TTA (original + hflip) per member, average results
6. Argmax for final prediction

---

## 📊 Ablation Study Results

| Approach | Val mIoU | Improvement vs Baseline | Key Finding |
|----------|----------|------------------------|-------------|
| Baseline | ~0.35 | — | Starting point |
| Member 1: Fine-Tuning | **0.4974** | +14% | Unfreezing backbone helps significantly |
| Member 2: Augmentation | ~0.47 | +12% | Data diversity improves rare classes |
| Member 3: Hyperparams | 0.4674 | +11% | OneCycleLR outperforms cosine |
| **Final Ensemble** | TBD | — | Combining all 3 approaches |

---

## 🎨 Class Mapping

16-bit mask values → class IDs:

| Raw Value | Class ID | Class Name | Color |
|-----------|----------|------------|-------|
| 0 | 0 | Background | Black |
| 100 | 1 | Trees | Forest Green |
| 200 | 2 | Lush Bushes | Lime |
| 300 | 3 | Dry Grass | Tan |
| 500 | 4 | Dry Bushes | Brown |
| 550 | 5 | Ground Clutter | Olive |
| 700 | 6 | Logs | Saddle Brown |
| 800 | 7 | Rocks | Gray |
| 7100 | 8 | Landscape | Sienna |
| 10000 | 9 | Sky | Sky Blue |

**Note:** Test set does not contain classes 6 (Logs) and some rare classes.

---

## ⚙️ Training Configuration

### Member 1 (Fine-Tuning)
- **Phase 1:** Frozen backbone, 25 epochs, batch=8, lr=3e-4
- **Phase 2:** Unfreeze last 4 blocks, 25 epochs, batch=4
  - Head LR: 3e-5
  - Backbone LR: 3e-6
- **Scheduler:** CosineAnnealingWarmRestarts(T_0=8, T_mult=2)
- **Loss:** Phased (CE+Dice → Focal+Dice)
- **Best mIoU:** 0.4974 (epoch 19, Phase 2)

### Member 2 (Augmentation)
- **Epochs:** 50
- **Batch:** 8
- **LR:** 1e-4 with CosineAnnealing
- **Augmentation:** 12 transforms (HFlip, SSR, RRC, Perspective, ColorJitter, Blur, Noise, Shadow, Fog, BC, Dropout, ToGray)
- **Class Weights:** [1, 2, 2, 1.5, 2, 3, 8, 3, 1, 1]
- **Loss:** 0.5×WeightedCE + 0.5×Dice

### Member 3 (Hyperparameters)
Three experiments:
1. `exp1_cosine_lr1e4` — CosineAnnealingLR, lr=1e-4
2. `exp2_cosine_lr5e5` — CosineAnnealingLR, lr=5e-5
3. `exp3_onecycle_lr1e4` — OneCycleLR, lr=1e-4 ← **Best: 0.4674**

---

## 📝 Important Notes

1. **TTA Scale Variants Disabled:** Scale variants (0.75×, 1.25×) break the fixed 18×33 token grid and were confirmed to hurt performance in Member 3's experiments. Only 2-variant TTA (original + hflip) is used.

2. **Loss Function:** Lovász-Softmax was planned but Dice was used as a fallback due to stability issues. This is documented in the Emergency Fallbacks.

3. **No Flowers Class:** Class 600 (Flowers) was in the plan but not present in actual training data. Final model uses 10 classes.

4. **CPU Inference:** Expected ~30-60 seconds per image on CPU. GPU recommended for batch processing.

---

## 📦 Files Included

```
submission/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── ensemble_all3.py                   # 3-model ensemble (Kaggle-ready)
├── tta.py                            # Standalone TTA inference
├── visualize.py                      # Visualization utilities
├── train.py                          # Training pipeline (Member 1)
├── member1_finetuning.py             # Member 1 notebook
├── member2_augmentation.py           # Member 2 notebook
├── member3_hyperparams_inference.py  # Member 3 notebook
├── model_finetuned_best.pth          # Member 1 weights
├── model_augmented_best.pth          # Member 2 weights
├── model_hypertuned_best.pth         # Member 3 weights
├── dataset.py                        # Dataset classes
├── augmentations.py                  # Augmentation pipeline
├── rare_class_tools.py               # Copy-paste augmentation
├── losses.py                         # Loss functions
├── audit_dataset.py                  # Dataset analysis
├── predictions/                      # Test set outputs
│   ├── masks/                        # 1,002 predictions
│   └── masks_color/                  # Colorized versions
├── train_stats/                      # Training logs and curves
└── REPORT.pdf                        # 8-page technical report
```

---

## 🔬 Reproducibility

All seeds are locked:
```python
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
```

---

## 📧 Contact

For issues or questions about this submission, refer to:
- `cross_verification_audit.md` — Detailed build audit
- `BUILD_CHECKLIST_UPDATED.md` — Original planning document
- `REPORT.pdf` — Full technical report (8 pages)

---

**Team:** Duality AI Hackathon  
**Submission Date:** April 2026  
**Best Single Model mIoU:** 0.4974 (Member 1)  
**Target Ensemble mIoU:** 0.70–0.80
