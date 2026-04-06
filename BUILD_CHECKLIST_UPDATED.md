# 🏗️ BUILD CHECKLIST — Duality AI Hackathon
## Offroad Semantic Scene Segmentation — Full Team Plan

> 3 Members · 3 Kaggle Notebooks · 1 Winning Submission
>
> **Rule**: Every checkbox must be ✅ before submission. No exceptions.

---

# 📖 HOW THIS PLAN WORKS

We use an **Ablation Study** strategy — each member trains the same base model with a **different improvement technique**. We then compare all 3 results scientifically and combine the best approach into one final model. This makes our report extremely strong because we can *prove* what helped and by how much.

```
Member 1 → Fine-Tuning (unfreeze DINOv2 backbone)
Member 2 → Data Augmentation + Class Weights (frozen backbone)
Member 3 → Hyperparameter Experiments (schedulers, LR tuning)
                    ↓
         Compare all 3 IoU scores
                    ↓
      Combine best approach → Final Model
                    ↓
               SUBMIT ✅
```

---

## 📋 Quick Reference

| Fact | Value |
|---|---|
| Train images | 2,857 (960×540 PNG) |
| Val images | 317 |
| Test images | 1,002 |
| Mask format | 16-bit PNG (`I;16`) |
| Classes | 10 (IDs: 0,100,200,300,500,550,600,700,800,7100,10000) |
| Rarest class | Logs — 0.07% of pixels |
| Test set missing | Flowers, Logs, Ground Clutter (only 7 classes) |
| Backbone | DINOv2 (provided: Small / upgrade: Base) |
| Scoring | 80pts IoU + 20pts Report |
| Platform | Kaggle (T4 GPU, 30hrs/week, batch≤8) |

---

## 👥 MEMBER ASSIGNMENTS

```
┌─────────────────────────────────────────────────────────────────────┐
│  MEMBER 1 (Fine-Tuning + Lead Engineer)                             │
│  Strategy: Unfreeze DINOv2 backbone — biggest expected IoU jump     │
│  Also builds: Core training pipeline + primary submission model     │
│  Files: train.py, losses.py, config.yaml                            │
├─────────────────────────────────────────────────────────────────────┤
│  MEMBER 2 (Augmentation + Data & Report)                            │
│  Strategy: Augmentation pipeline + class weights — best for         │
│  generalization to unseen desert locations                           │
│  Also builds: Dataset tooling + ablation baseline + full report     │
│  Files: dataset.py, augmentations.py, rare_class_tools.py,          │
│         audit_dataset.py, REPORT.pdf                                │
├─────────────────────────────────────────────────────────────────────┤
│  MEMBER 3 (Hyperparameter Tuning + Inference & Packaging)           │
│  Strategy: Systematic LR/scheduler experiments — best polish        │
│  Also builds: Test pipeline + TTA + backbone experiment + README    │
│  Files: test.py, tta.py, visualize.py, README.md,                   │
│         requirements.txt                                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 3-APPROACH TRAINING STRATEGY

### Why 3 Separate Approaches?

Each member runs their own Kaggle notebook simultaneously. We compare IoU scores at the end and pick what works. This is exactly how professional ML teams operate.

| Approach | Member | Expected Val IoU | What it proves |
|----------|--------|-----------------|----------------|
| Baseline (no changes, 10 epochs) | — | 0.30 – 0.40 | Starting point |
| Fine-Tuning only | Member 1 | 0.55 – 0.65 | How much unfreezing backbone helps |
| Augmentation + Class Weights only | Member 2 | 0.50 – 0.65 | How much data diversity helps |
| Hyperparameter Tuning only | Member 3 | 0.45 – 0.55 | How much settings matter |
| **All combined (final submission)** | All 3 | **0.70 – 0.80** | **Best possible model** |

### Ablation Results Table *(fill in during Phase 2)*

| Approach | Val IoU | Improvement vs Baseline | Key Finding |
|----------|---------|------------------------|-------------|
| Baseline | ___ | — | Starting point |
| Member 1: Fine-Tuning | ___ | ___ | |
| Member 2: Augmentation | ___ | ___ | |
| Member 3: Hyperparams | ___ | ___ | |
| **Final Combined** | ___ | ___ | **Submitted model** |

### Decision Rule After All 3 Finish

```
Compare all 3 IoU scores
→ Best single approach becomes the base for final model
→ Add remaining useful techniques on top
→ Train one final combined model
→ That is what gets submitted
```

---

## 🔴 PHASE 0: Environment Setup (ALL Members — Day 1, first 30 min)

Every member does these steps on their own Kaggle notebook:

- [ ] Create Kaggle account (if not done)
- [ ] Create new notebook → enable GPU T4 → enable Internet
- [ ] Upload dataset ZIP to Kaggle Datasets
- [ ] Verify dataset paths:
  ```
  /kaggle/input/YOUR-DATASET/Offroad_Segmentation_Training_Dataset/train/Color_Images/
  /kaggle/input/YOUR-DATASET/Offroad_Segmentation_Training_Dataset/train/Segmentation/
  /kaggle/input/YOUR-DATASET/Offroad_Segmentation_Training_Dataset/val/Color_Images/
  /kaggle/input/YOUR-DATASET/Offroad_Segmentation_Training_Dataset/val/Segmentation/
  /kaggle/input/YOUR-DATASET/Offroad_Segmentation_testImages/Color_Images/
  /kaggle/input/YOUR-DATASET/Offroad_Segmentation_testImages/Segmentation/
  ```
- [ ] Run `!pip install albumentations` in notebook
- [ ] Verify GPU is active: `!nvidia-smi`
- [ ] Verify PyTorch sees GPU: `torch.cuda.is_available()` → `True`
- [ ] Verify DINOv2 downloads: `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")` → no error
- [ ] Verify image loads: `Image.open(sample_path).size` → `(960, 540)`
- [ ] Verify mask loads: `np.array(Image.open(mask_path)).dtype` → `uint16`
- [ ] Verify mask values: `np.unique(...)` → `[100, 200, 300, 500, 550, ...]`

**✅ GATE: All 3 members confirm environment works before proceeding.**

---

## 🟡 PHASE 1: Core Code Build (Day 1 — parallel work)

---

### MEMBER 1 — Fine-Tuning + Core Training Pipeline

> **Training Strategy**: Unfreeze DINOv2 backbone so the entire network learns from desert data. Use two-phase training — frozen first for stability, then unfreeze for fine-tuning.

#### 1A. Config File (`config.yaml`)
- [ ] Create `config.yaml` with all hyperparameters:
  ```yaml
  # Model
  backbone: "small"      # "small" or "base"
  n_classes: 10
  image_h: 252           # (540/2) rounded to ×14
  image_w: 462           # (960/2) rounded to ×14

  # Training Phase 1 (frozen backbone)
  phase1_epochs: 25
  phase1_batch_size: 8
  phase1_lr: 3e-4

  # Training Phase 2 (unfrozen last 4 blocks)
  phase2_epochs: 25
  phase2_batch_size: 4
  phase2_head_lr: 3e-5
  phase2_backbone_lr: 3e-6

  # Optimizer
  optimizer: "adamw"
  weight_decay: 0.01

  # Scheduler
  scheduler: "cosine_warm_restarts"
  T_0: 8
  T_mult: 2
  eta_min: 1e-6
  warmup_steps: 300

  # Regularization
  gradient_clip: 1.0
  label_smoothing: 0.0
  dropout: 0.1
  amp: true

  # Early stopping
  patience: 12

  # Paths (Kaggle)
  train_dir: "/kaggle/input/..."
  val_dir: "/kaggle/input/..."
  output_dir: "/kaggle/working/"
  ```
- [ ] Config loads correctly in Python (`yaml.safe_load`)

#### 1B. Key Code Changes for Fine-Tuning

```python
# Step 1: Load backbone normally first (frozen)
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
backbone.eval()
for param in backbone.parameters():
    param.requires_grad = False   # frozen for Phase 1

# Step 2: Phase 1 optimizer — only train the head
optimizer = optim.AdamW(classifier.parameters(), lr=3e-4, weight_decay=0.01)

# Step 3: After Phase 1 (epoch 25), unfreeze last 4 backbone blocks
for block in backbone.blocks[-4:]:
    for param in block.parameters():
        param.requires_grad = True

# Step 4: Phase 2 optimizer — dual learning rates
optimizer = optim.AdamW([
    {'params': classifier.parameters(),          'lr': 3e-5},
    {'params': backbone.blocks[-4:].parameters(), 'lr': 3e-6},  # much lower!
], weight_decay=0.01)

# Step 5: Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=8, T_mult=2, eta_min=1e-6
)
```

#### 1C. Loss Functions (`losses.py`)
- [ ] **Lovász-Softmax loss** implementation
  - [ ] Verify: `lovasz_softmax(F.softmax(logits, dim=1), targets)` runs without error
  - [ ] Verify: returns a scalar tensor with `.backward()` working
- [ ] **Dice loss** implementation
  - [ ] Per-class Dice computation, smooth factor = 1e-6, returns `1 - mean_dice`
- [ ] **Focal loss** implementation
  - [ ] Accepts `gamma` and `alpha` parameters, default `gamma=2.0`
- [ ] **Phased loss function** — different loss per training stage:
  - Epochs 1–15: `0.5 × WeightedCE + 0.5 × Dice`
  - Epochs 15–35: `0.3 × WeightedCE + 0.7 × Lovász`
  - Epochs 35–50: `0.2 × Focal + 0.8 × Lovász`

#### 1D. Enhanced Decoder
- [ ] `SegmentationHeadConvNeXt` with Stem → Block1 → Block2 → Dropout(0.1) → Classifier
- [ ] Verify: forward pass produces correct output shape

#### 1E. Training Loop (`train.py`)
- [ ] AdamW optimizer (not SGD)
- [ ] CosineAnnealingWarmRestarts scheduler
- [ ] Linear warmup first 300 steps
- [ ] AMP (FP16): `torch.cuda.amp.autocast()` + `GradScaler`
- [ ] Gradient clipping: `torch.nn.utils.clip_grad_norm_(params, 1.0)`
- [ ] Phase 1: frozen backbone, batch size=8, epochs 1–25
- [ ] Phase 2: unfreeze last 4 blocks, batch size=4, epochs 26–50
- [ ] Phased loss switching per epoch
- [ ] Early stopping: patience=12
- [ ] Checkpointing: best by val mIoU + every 10 epochs
- [ ] Save: `model_finetuned_best.pth`, training curves, metrics
- [ ] Verify: training runs 3 epochs without crash

**Expected output:** `model_finetuned_best.pth` — Expected IoU: **+15% to +25%** over baseline

---

### MEMBER 2 — Augmentation + Class Weights + Dataset Tools

> **Training Strategy**: Keep backbone fully frozen. Focus on making the model see MORE variety through augmentation and pay attention to rare classes through weighted loss. This directly addresses the domain shift problem (train→test different location).

#### 2A. Dataset Audit (`audit_dataset.py`)
- [ ] Count images per split (train/val/test)
- [ ] Compute pixel-level class distribution from ALL train masks
- [ ] Generate class distribution bar chart → `class_distribution.png`
- [ ] **KEY OUTPUT**: computed class weights (inverse sqrt frequency):
  ```python
  weights = 1.0 / torch.sqrt(pixel_frequencies + 1e-3)
  weights = weights / weights.sum() * num_classes
  ```
- [ ] Save weights to file that `train.py` can load

#### 2B. Augmentation Pipeline (`augmentations.py`)

> These apply **automatically during training** — images on disk are never modified. Every time the model sees an image it gets a randomly transformed version.

- [ ] **Albumentations train transform** (12 transforms):
  - [ ] `HorizontalFlip(p=0.5)` — ❌ NO vertical flip
  - [ ] `ShiftScaleRotate(shift=0.1, scale=0.3, rotate=15, p=0.7)`
  - [ ] `RandomResizedCrop(h=252, w=462, scale=(0.5,1.5), p=1.0)`
  - [ ] `Perspective(scale=(0.02, 0.05), p=0.2)`
  - [ ] `OneOf([ColorJitter, RandomGamma, CLAHE], p=0.5)`
  - [ ] `OneOf([GaussianBlur, MotionBlur], p=0.2)`
  - [ ] `GaussNoise(var_limit=(10,50), p=0.15)`
  - [ ] `RandomShadow(p=0.2)`
  - [ ] `RandomFog(fog_coef=(0.1,0.25), p=0.1)` ← simulates desert haze
  - [ ] `RandomBrightnessContrast(p=0.3)` ← simulates different times of day
  - [ ] `CoarseDropout(max_holes=6, max_h=40, max_w=40, p=0.15)`
  - [ ] `ToGray(p=0.05)`
  - [ ] `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`
  - [ ] `ToTensorV2()`
- [ ] **Val/Test transform** (no augmentation — just resize+normalize+tensor)
- [ ] **Visual verification**: save 5 augmented image+mask pairs — confirm mask alignment

#### 2C. Class Weights for Loss Function

> This is the most important fix for rare classes. Without this, the model ignores Logs and Flowers entirely.

```python
# Computed from real data (use audit_dataset.py output)
# Higher weight = model pays more attention to that class
class_weights = torch.tensor([
    1.0,   # Background  — very common, low weight
    2.0,   # Trees
    2.0,   # Lush Bushes
    1.5,   # Dry Grass
    2.0,   # Dry Bushes
    3.0,   # Ground Clutter — rare
    5.0,   # Logs           — very rare (0.07% of pixels!)
    3.0,   # Rocks
    1.0,   # Landscape     — very common, low weight
    1.0,   # Sky           — very common, low weight
]).to(device)

loss_fct = nn.CrossEntropyLoss(weight=class_weights)
```

#### 2D. Rare Class Tools (`rare_class_tools.py`)
- [ ] `build_rare_class_pool()` — scans all training masks, extracts crops of Logs/Flowers
- [ ] `copy_paste_augment()` — pastes rare class crops at random positions in batch
- [ ] `WeightedRandomSampler` — 3x weight for images containing Logs/Flowers
- [ ] Verify: visualize 5 copy-paste results — mask alignment correct

#### 2E. Ablation Baseline Training (Kaggle Notebook 2)

Minimal changes from original script — just enough to be competitive:

- [ ] SGD → AdamW (`lr=1e-4, weight_decay=0.01`)
- [ ] Add computed class weights to `CrossEntropyLoss(weight=...)`
- [ ] Epochs: 10 → 50
- [ ] Add `CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)`
- [ ] Add full augmentation pipeline from `augmentations.py`
- [ ] **NO backbone unfreezing, NO Lovász, NO decoder changes**
- [ ] Save: `model_augmented_best.pth`, `augmentation_metrics.txt`, `augmentation_curves.png`

**Expected output:** `model_augmented_best.pth` — Expected IoU: **+20% to +30%** especially on rare classes

#### 2F. Report Writing (`REPORT.md` → `REPORT.pdf`)
- [ ] Page 1: Title + team name + one-paragraph summary of approach and best mIoU
- [ ] Page 2: Methodology — architecture diagram, two-phase training, loss design
- [ ] Page 3: Data Analysis — class distribution chart, train vs test comparison, augmentation list
- [ ] Page 4: **Ablation Study Results** — comparison table of all 3 approaches with per-class IoU
- [ ] Page 5: Per-Class Analysis — confusion matrix, side-by-side GT vs prediction images
- [ ] Page 6: Challenges & Solutions
- [ ] Page 7: Failure Cases — 3-4 images where model fails with analysis of why
- [ ] Page 8: Conclusion & Future Work
- [ ] Convert to PDF ≤ 8 pages, file name: `REPORT.pdf`

---

### MEMBER 3 — Hyperparameter Experiments + Inference & Packaging

> **Training Strategy**: Run systematic experiments to find the best learning rate and scheduler combination. Uses early stopping so bad configs are killed quickly. Identifies the best settings for the final combined model.

#### 3A. Hyperparameter Experiments (Kaggle Notebook 3)

Run these 3 experiments back to back and log results:

```python
EXPERIMENTS = [
    {
        "name": "exp1_cosine_lr1e4",
        "lr": 1e-4,
        "scheduler": "cosine",     # CosineAnnealingLR
        "n_epochs": 50,
        "early_stop_patience": 10,
    },
    {
        "name": "exp2_cosine_lr5e5",
        "lr": 5e-5,                # lower learning rate
        "scheduler": "cosine",
        "n_epochs": 50,
        "early_stop_patience": 10,
    },
    {
        "name": "exp3_onecycle_lr1e4",
        "lr": 1e-4,
        "scheduler": "onecycle",   # OneCycleLR — different shape
        "n_epochs": 50,
        "early_stop_patience": 10,
    },
]
```

- [ ] All 3 experiments start from same base code (frozen backbone, AdamW, no augmentation)
- [ ] Each experiment saves: `model_best.pth`, per-epoch metrics CSV, loss curve
- [ ] Print comparison table at end showing which config won
- [ ] Identify: best LR, best scheduler — report these to team

**Expected output:** Best settings identified — Expected IoU: **+5% to +15%** over baseline

#### 3B. Key Code for Experiments

```python
# Cosine scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=n_epochs, eta_min=1e-6
)

# OneCycle scheduler (steps per batch not per epoch)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=lr,
    steps_per_epoch=len(train_loader),
    epochs=n_epochs
)

# Early stopping
patience_counter = 0
if val_iou > best_val_iou:
    best_val_iou = val_iou
    patience_counter = 0
    torch.save(classifier.state_dict(), f"{exp_dir}/model_best.pth")
else:
    patience_counter += 1
    if patience_counter >= early_stop_patience:
        print(f"Early stopping at epoch {epoch+1}")
        break
```

#### 3C. Test Script with TTA (`test.py`)
- [ ] `--tta` flag to enable/disable Test-Time Augmentation
- [ ] 4 TTA variants: original, horizontal flip, scale 0.75×, scale 1.25×
- [ ] Average softmax probabilities across variants → argmax
- [ ] Works on val set (with GT masks → computes metrics)
- [ ] Works on test set (saves predictions for all 1,002 images)
- [ ] Per-class IoU reporting (not just mean):
  ```
  Background:     0.0000
  Trees:          0.XXXX
  Logs:           0.XXXX   ← watch this one
  ...
  Mean IoU:       0.XXXX
  ```
- [ ] Save: `model_hypertuned_best.pth`, `hypertuned_metrics.txt`

#### 3D. Visualization Improvements (`visualize.py`)
- [ ] Confusion matrix heatmap (`sklearn.metrics.confusion_matrix`) → `confusion_matrix.png`
- [ ] Per-class IoU bar chart with colors (green=good, red=bad)
- [ ] Failure case gallery — 5 images with LOWEST IoU, side-by-side: Input | GT | Prediction
- [ ] Augmentation showcase — 1 original + 6 augmented versions with masks (for report)

#### 3E. Packaging & README
- [ ] `requirements.txt` with all dependencies
- [ ] `README.md` with step-by-step instructions:
  ```bash
  # Install
  pip install -r requirements.txt

  # Train
  python train.py --config config.yaml

  # Test with TTA
  python test.py --model_path weights/best_model.pth --data_dir path/to/test --tta
  ```
- [ ] Submission folder structure verified (see Phase 4)

---

## 🟢 PHASE 2: Training Runs (Days 2–3)

### All Members — Run Simultaneously on Kaggle

> All 3 run at the same time on separate Kaggle accounts. No need to wait for each other.

| Notebook | Member | What's Training | Expected Time | Expected mIoU |
|---|---|---|---|---|
| Kaggle 1 | Member 1 | Phase 1: frozen backbone, 25 epochs | 2–4 hours | 0.45–0.55 |
| Kaggle 2 | Member 2 | Augmentation + class weights, 50 epochs | 2–3 hours | 0.50–0.65 |
| Kaggle 3 | Member 3 | 3 hyperparameter experiments, 50 epochs each | 3–5 hours | 0.45–0.55 |

**Checkpoints**: save to `/kaggle/working/` every 5 epochs. Download `.pth` files after each run.

#### After Phase 1 (Member 1 only):
- [ ] Check val mIoU — is it > 0.40? If not, debug before Phase 2
- [ ] Check per-class IoU — which classes are near zero?
- [ ] If Logs IoU ≈ 0: activate copy-paste augmentation for Phase 2
- [ ] Start Phase 2 (unfreeze last 4 blocks, continue 25 more epochs)

#### After ALL Training Runs Complete:
- [ ] **Member 1** reports final val mIoU: ___
- [ ] **Member 2** reports final val mIoU: ___
- [ ] **Member 3** reports best experiment mIoU: ___ (which experiment: ___)
- [ ] Fill in ablation table with real numbers
- [ ] **DECISION**:
  - [ ] Pick the best single approach as base
  - [ ] Add best elements from other approaches on top
  - [ ] One member trains the final combined model
  - [ ] Final model is what gets submitted

---

## 🔵 PHASE 3: Test Predictions + TTA (Day 3–4)

- [ ] Load best model weights
- [ ] Run `test.py --tta` on **validation set** first → record mIoU with TTA
- [ ] Compare: mIoU without TTA vs with TTA → record the improvement
- [ ] Run `test.py --tta` on **test set** → save all prediction masks
- [ ] Verify predictions visually — spot-check 10 random test images:
  - [ ] Does Sky look right?
  - [ ] Are Rocks being detected? (17.7% of test data — must be accurate)
  - [ ] Is model hallucinating Flowers/Logs on test data? (they shouldn't appear)
- [ ] If hallucination problem: add post-processing confidence threshold for rare classes
- [ ] Verify: exactly 1,002 prediction files saved
- [ ] Save all outputs to `predictions/` folder

---

## 🟣 PHASE 4: Report + Final Packaging (Day 4–5)

### Member 2 — Report Finalization
- [ ] Insert real numbers into ablation study table (all 3 approaches + final combined)
- [ ] Insert real training curves from all 3 experiments (overlay them on same graph)
- [ ] Insert real confusion matrix from best model
- [ ] Insert real failure case images
- [ ] Proofread all sections
- [ ] Export to PDF ≤ 8 pages
- [ ] File name: `REPORT.pdf`

### Member 3 — Final Packaging
- [ ] All files in submission folder (see structure below)
- [ ] `requirements.txt` complete and tested
- [ ] `README.md` clear enough for someone unfamiliar
- [ ] Model weights included (`.pth` file)
- [ ] Predictions included (`predictions/` folder)
- [ ] **Dry run test**: fresh Kaggle notebook, install deps, load model, run test.py → works?

### Member 1 — Final Review
- [ ] Review all code for errors
- [ ] Verify config matches what was actually trained
- [ ] Verify model weights match the architecture in code
- [ ] Verify class mapping is IDENTICAL across ALL files:
  ```python
  # THIS MUST BE THE SAME in train.py, test.py, dataset.py
  value_map = {
      0:     0,   # Background
      100:   1,   # Trees
      200:   2,   # Lush Bushes
      300:   3,   # Dry Grass
      500:   4,   # Dry Bushes
      550:   5,   # Ground Clutter
      600:   6,   # Flowers   ← verify if present in actual masks
      700:   7,   # Logs
      800:   8,   # Rocks
      7100:  9,   # Landscape
      10000: 10,  # Sky
  }
  ```
  > ⚠️ The provided script maps 10 classes (0–9) but Flowers (600) appears in data. Run `audit_dataset.py` to verify actual classes before training!

---

## 🔶 CRITICAL VERIFICATION GATES

### Gate 1: Before ANY Training
- [ ] Augmented image + mask pair visually verified (mask aligns after augmentation)
- [ ] Loss function runs without NaN on a test batch
- [ ] Class weights computed from real data (not hardcoded)
- [ ] Data loader produces correct shapes: `images=[B,3,252,462]`, `masks=[B,252,462]`
- [ ] Mask values are 0–9 integers (not raw 100/200/300... values)

### Gate 2: After First 5 Epochs
- [ ] Loss is decreasing (not flat, not NaN, not increasing)
- [ ] Val mIoU > 0.10 (model is learning something)
- [ ] GPU utilization > 80% (not CPU bottlenecked)
- [ ] No CUDA OOM errors

### Gate 3: Before Submission
- [ ] Best model weights file exists and loads without error
- [ ] `test.py` runs on test set and produces predictions for ALL 1,002 images
- [ ] Predictions folder has exactly 1,002 mask files
- [ ] Report PDF is ≤ 8 pages
- [ ] README has working commands
- [ ] Class mapping consistent across ALL Python files
- [ ] No test images were used in training (**instant DQ if violated**)

---

## 📁 SUBMISSION FOLDER STRUCTURE

```
submission/
├── README.md
├── requirements.txt
├── config.yaml
├── train.py
├── test.py
├── losses.py
├── augmentations.py
├── dataset.py
├── rare_class_tools.py
├── tta.py
├── visualize.py
├── audit_dataset.py
├── segmentation_head.pth       ← best model weights
├── train_stats/
│   ├── training_curves.png
│   ├── iou_curves.png
│   └── evaluation_metrics.txt
├── predictions/
│   ├── masks/                  ← 1,002 raw prediction masks
│   ├── masks_color/            ← 1,002 colorized masks
│   └── comparisons/            ← side-by-side comparison images
└── REPORT.pdf
```

---

## 📎 CODE DEPENDENCIES MAP

```
config.yaml
    ↓
train.py ──────→ dataset.py (data loading)
    │              ↓
    │         augmentations.py (transforms)
    │              ↓
    │         rare_class_tools.py (copy-paste pool)
    │
    ├──→ losses.py (Lovász, Dice, Focal, phased)
    │
    └──→ saves: segmentation_head.pth
                    ↓
               test.py ──→ tta.py (test-time augmentation)
                    │
                    └──→ saves: predictions/masks/
                                predictions/masks_color/
                                evaluation_metrics.txt
                                       ↓
                                  visualize.py ──→ confusion_matrix.png
                                                   failure_cases.png
                                                        ↓
                                                   REPORT.pdf
```

**Build order**: `config.yaml` → `losses.py` → `augmentations.py` → `dataset.py` → `rare_class_tools.py` → `train.py` → `tta.py` → `test.py` → `visualize.py`

---

## ⏰ DAILY SYNC CHECKPOINTS

| Time | What to Confirm |
|---|---|
| **Day 1 End** | All 3 members: environment works, code compiles, can run 1 training epoch |
| **Day 2 End** | Member 1: Phase 1 mIoU = ___. Member 2: Augmentation training running. Member 3: First experiment done. |
| **Day 3 End** | All training done. Best model selected. Ablation table filled. TTA tested. |
| **Day 4 End** | Test predictions generated. Report 80% done. README drafted. |
| **Day 5 End** | Everything packaged. Dry-run verified. **SUBMIT.** |

---

## 🚨 EMERGENCY FALLBACKS

| Problem | Fallback |
|---|---|
| DINOv2-Base OOMs | Use DINOv2-Small (already in provided script) |
| Lovász causes NaN | Use `WeightedCE + Dice` only (skip Lovász entirely) |
| Backbone unfreeze OOMs | Keep backbone 100% frozen. Member 2's augmentation approach becomes primary. |
| Augmentation corrupts masks | Disable Albumentations. Use only `HorizontalFlip` via torchvision. |
| Val mIoU stuck < 0.35 | Check class mapping. Check mask normalization (`*255` in original script). Check AdamW not SGD. |
| Kaggle session dies | Always checkpoint. Build `--resume` flag into train.py. |
| TTA dimension mismatch | Skip scale variants. Use only original + horizontal flip (2 variants). |
| Report too long | Cut Failure Cases to 1 page. Cut Future Work to 3 sentences. |
| Can't reproduce results | Lock `seed=42` in torch, numpy, random, CUDA. Set `deterministic=True`. |
| Member 3 experiments take too long | Run only exp1 and exp2. Skip exp3. Share results after exp1 done. |
| All 3 IoU scores similar | Combine ALL techniques into final model — every bit helps. |

---

## 📊 IoU SCORE REFERENCE

| IoU Score | Meaning | Action |
|-----------|---------|--------|
| 0.0 – 0.3 | Very Poor | Something is wrong — check data paths and class mapping |
| 0.3 – 0.5 | Poor / Baseline | Normal starting point, apply improvements |
| 0.5 – 0.65 | Decent | Model is learning well |
| 0.65 – 0.75 | Good | Competitive submission |
| 0.75 – 0.85 | Very Good | Strong submission, likely top ranks |
| 0.85+ | Excellent | Top tier |

> **Target**: Final combined model Val IoU **above 0.70** before submitting.

---

> **🏁 "The model that wins isn't the most complex. It's the one trained with the right data, the right loss, on the right hardware, submitted on time."**
