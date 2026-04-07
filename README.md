# DINOv2 Offroad Semantic Segmentation - M3 Model

**Duality AI Hackathon Submission** — Best performing single model for desert terrain segmentation.

---

## 📋 Quick Reference

| Metric | Value |
|--------|-------|
| **Current Best Model** | **Member 3 (M3)** - Hyperparameter Tuned |
| Backbone | DINOv2 vits14 (384-dim) |
| Image Size | 252×462 (18×33 token grid) |
| Classes | 10 (Background, Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Logs, Rocks, Landscape, Sky) |
| Decoder | SegmentationHeadConvNeXt (5.19M params) |
| **Validation mIoU** | **0.4158 (41.58%)** |
| TTA | 2-variant (original + horizontal flip) |

---

## ⚠️ Important Update (April 7, 2026)

### Current Status
- **M3 is the best working model** with 0.4158 mIoU
- **M1 (0.4974 mIoU)** has Windows loading issues (PyTorch directory format)
- **M2** achieves ~0.37 mIoU
- **Weight merging approach failed** - averaging model weights destroys performance
- **Soft voting ensemble** is the correct approach but requires all 3 models

### Known Issues
1. **Background class (0)** has 0 IoU - models never predict it
2. **Rare classes** (Logs, Rocks, Ground Clutter) have low IoU
3. **Sky class dominates** predictions, inflating pixel accuracy

---

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/Adity-ng/offroad-segmentation.git
cd offroad-segmentation

# Install dependencies
pip install torch torchvision pillow numpy tqdm

# Download DINOv2 (automatic on first run)
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+ (CPU works, GPU recommended)
- 8GB+ RAM
- ~20GB disk space for models

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

## 📁 Model Files

| File | Description | Size |
|------|-------------|------|
| `model_member3_new.pth` | **M3 - Best Model** | ~20MB |
| `model_augmented_best.pth` | M2 - Augmented | ~20MB |
| `model_finetuned_best.pth.zip` | M1 - Fine-tuned (zipped) | ~70MB |

---

## 🎯 Quick Start

### Generate Test Predictions with M3

```bash
python proper_ensemble.py
```

This will:
1. Load M3 model
2. Run validation (317 images)
3. Generate test predictions (1,002 images)
4. Save to `final_ensemble_predictions/`

### Single Image Inference

```python
import torch
from PIL import Image
import torchvision.transforms as transforms

# Load model
model = SegmentationHeadConvNeXt(384, 10, 33, 18)
model.load_state_dict(torch.load('model_member3_new.pth'))
model.eval()

# Load backbone
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
backbone.eval()

# Inference
img = Image.open('test.png').convert('RGB')
img_tensor = transforms.Compose([
    transforms.Resize((252, 462)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])(img)

with torch.no_grad():
    feats = backbone.forward_features(img_tensor.unsqueeze(0))['x_norm_patchtokens']
    logits = model(feats)
    pred = torch.argmax(logits, dim=1)
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

## 📊 Model Comparison

| Model | Strategy | Val mIoU | Status |
|-------|----------|----------|--------|
| **M3** | OneCycleLR + Hyperparams | **0.4158** | ✅ **Working** |
| M2 | Augmentation + Class Weights | 0.3703 | ✅ Working |
| M1 | Fine-tuning (unfreeze backbone) | 0.4974 | ⚠️ Windows loading issue |
| **Merged Weights** | Average of M2+M3 | 0.2194 | ❌ Failed |

### M3 Per-Class Performance

| Class | IoU | Status |
|-------|-----|--------|
| Sky | 0.9843 | ✅ Excellent |
| Landscape | 0.5665 | ✅ Good |
| Dry Grass | 0.6771 | ✅ Good |
| Dry Bushes | 0.3657 | ⚠️ Moderate |
| Rocks | 0.3278 | ⚠️ Moderate |
| Lush Bushes | 0.2033 | ❌ Poor |
| Trees | 0.1205 | ❌ Poor |
| Ground Clutter | 0.0815 | ❌ Poor |
| Logs | 0.0000 | ❌ Missing |
| Background | N/A | ❌ Never predicted |

---

## 📊 Ablation Study Results

| Approach | Val mIoU | Status | Key Finding |
|----------|----------|--------|-------------|
| **M3 (OneCycleLR)** | **0.4158** | ✅ **Best Working** | Hyperparam tuning helps |
| M2 (Augmentation) | 0.3703 | ✅ Working | Augmentation diversity |
| M1 (Fine-Tuning) | 0.4974 | ⚠️ Load Issue | Backbone fine-tuning is best |
| Weight Merging | 0.2194 | ❌ Failed | Don't average weights |
| **Target** | 0.50+ | 🎯 | Need M1 loaded properly |

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

## 📦 Repository Structure

```
offroad-segmentation/
├── README.md                          # This file
├── model_member3_new.pth              # ⭐ M3 - Best working model (20MB)
├── model_augmented_best.pth           # M2 - Augmented model (20MB)
├── model_finetuned_best.pth.zip       # M1 - Best performer (70MB zipped)
├── proper_ensemble.py                 # ⭐ Main inference script
├── validate_merged_model.py           # Validation script
├── create_merged_model_v2.py          # Model merging (failed approach)
├── test_ensemble_3models.py           # Original ensemble script
├── train_high_performance.py          # Training script
├── Offroad_Segmentation_Training_Dataset/  # Training data
│   ├── train/
│   └── val/
├── Offroad_Segmentation_testImages/   # Test data
├── final_ensemble_predictions/        # Output predictions
└── merged_predictions/                # Previous run outputs
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

## 🚀 GitHub Push Commands

```bash
# Add M3 model with Git LFS
git lfs track "*.pth"
git add .gitattributes
git add model_member3_new.pth
git commit -m "Add M3 model - best performing (0.4158 mIoU)"

# Add code
git add README.md proper_ensemble.py validate_merged_model.py
git commit -m "Update README and add validation scripts"

# Push
git push origin main
```

---

## 📝 Important Notes

1. **M3 is current best model** for submission
2. **Predictions saved as uint16 PNG** with correct mask values (0, 100, 200, etc.)
3. **Background issue** - models never predict class 0; may need retraining with balanced data
4. **CPU inference** takes ~1-2 seconds per image

---

## 🔮 Future Improvements

1. **Fix M1 loading** on Linux to achieve 0.50 mIoU
2. **Retrain with class balancing** for rare classes
3. **Add background pixels** to training if missing
4. **Try DeepLabV3+ or U-Net** decoder alternatives
5. **Test-time augmentation with scales** (if token grid allows)

---

## 📧 Contact

**Team:** Duality AI Hackathon  
**Date:** April 2026  
**Best Model:** Member 3 (M3) - 0.4158 mIoU  
**Target:** Achieve 0.50+ mIoU with proper ensemble

---
