# 🏗️ BUILD CHECKLIST — Duality AI Hackathon

> 3 Members · 3 Laptops · 1 Winning Submission
>
> **Rule**: Every checkbox must be ✅ before submission. No exceptions.

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
┌─────────────────────────────────────────────────────────────────┐
│  MEMBER 1 (Lead Engineer)                                       │
│  Builds: Core training pipeline + primary submission model      │
│  Files: train.py, losses.py, augmentations.py, config.yaml      │
├─────────────────────────────────────────────────────────────────┤
│  MEMBER 2 (Data & Report)                                       │
│  Builds: Dataset tooling + ablation baseline + full report       │
│  Files: dataset.py, audit_dataset.py, rare_class_tools.py,      │
│         REPORT.pdf                                               │
├─────────────────────────────────────────────────────────────────┤
│  MEMBER 3 (Inference & Packaging)                               │
│  Builds: Test pipeline + TTA + backbone experiment + README      │
│  Files: test.py, tta.py, visualize.py, README.md,               │
│         requirements.txt                                         │
└─────────────────────────────────────────────────────────────────┘
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

### MEMBER 1 — Core Training Pipeline

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

#### 1B. Loss Functions (`losses.py`)
- [ ] **Lovász-Softmax loss** implementation
  - [ ] Copy from: https://github.com/bermanmb/lovern-losses (MIT license)
  - [ ] Verify: `lovasz_softmax(F.softmax(logits, dim=1), targets)` runs without error
  - [ ] Verify: returns a scalar tensor with `.backward()` working
- [ ] **Dice loss** implementation
  - [ ] Per-class Dice computation
  - [ ] Smooth factor = 1e-6
  - [ ] Returns `1 - mean_dice`
- [ ] **Focal loss** implementation
  - [ ] Accepts `gamma` and `alpha` (class weights) parameters
  - [ ] Default `gamma=2.0`
  - [ ] Verify: handles class weights correctly
- [ ] **Phased loss function**
  - [ ] `get_loss_fn(epoch, class_weights, device)` returns correct loss for each phase:
    - Epochs 1–15: `0.5 * WeightedCE + 0.5 * Dice`
    - Epochs 15–35: `0.3 * WeightedCE + 0.7 * Lovász`
    - Epochs 35–50: `0.2 * Focal + 0.8 * Lovász`
  - [ ] Unit test: call with epoch=5, 20, 40 and verify different loss values

#### 1C. Augmentation Pipeline (`augmentations.py`)
- [ ] **Albumentations train transform** (12 transforms):
  - [ ] `HorizontalFlip(p=0.5)` — ❌ NO vertical flip
  - [ ] `ShiftScaleRotate(shift=0.1, scale=0.3, rotate=15, p=0.7)`
  - [ ] `RandomResizedCrop(h=252, w=462, scale=(0.5,1.5), p=1.0)`
  - [ ] `Perspective(scale=(0.02, 0.05), p=0.2)`
  - [ ] `OneOf([ColorJitter, RandomGamma, CLAHE], p=0.5)`
  - [ ] `OneOf([GaussianBlur, MotionBlur], p=0.2)`
  - [ ] `GaussNoise(var_limit=(10,50), p=0.15)`
  - [ ] `RandomShadow(p=0.2)`
  - [ ] `RandomFog(fog_coef=(0.1,0.25), p=0.1)`
  - [ ] `RandomBrightnessContrast(p=0.3)`
  - [ ] `CoarseDropout(max_holes=6, max_h=40, max_w=40, p=0.15)`
  - [ ] `ToGray(p=0.05)`
  - [ ] `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`
  - [ ] `ToTensorV2()`
- [ ] **Val/Test transform** (no augmentation, just resize+normalize+tensor)
- [ ] **CutMix function** `cutmix_batch(images, masks, alpha=1.0)`
  - [ ] Returns modified images and masks
  - [ ] Verify: masks are correctly cut-and-pasted (not corrupted)
- [ ] **Copy-Paste function** `copy_paste_rare_classes(images, masks, pool)`
  - [ ] Accepts pre-built pool of rare class crops
  - [ ] Pastes crops at random positions
  - [ ] Masks updated correctly with pasted class IDs
- [ ] **Visual verification**: save 5 augmented image+mask pairs, visually confirm mask alignment

#### 1D. Enhanced Decoder (modify in `train.py`)
- [ ] New `SegmentationHeadConvNeXt` with:
  - [ ] Stem: `Conv2d(in_ch→256) + BN + GELU`
  - [ ] Block 1: depthwise-separable `Conv2d(256→256)` + BN + GELU + residual
  - [ ] Block 2: `Conv2d(256→128)` + BN + GELU
  - [ ] `Dropout2d(0.1)`
  - [ ] Classifier: `Conv2d(128→10)`
- [ ] Verify: forward pass produces correct output shape

#### 1E. Modified Training Loop (`train.py`)
- [ ] Load config from `config.yaml`
- [ ] **Optimizer**: AdamW (not SGD)
- [ ] **Scheduler**: CosineAnnealingWarmRestarts
- [ ] **Linear warmup**: first 300 steps
- [ ] **AMP (FP16)**: `torch.cuda.amp.autocast()` + `GradScaler`
- [ ] **Gradient clipping**: `torch.nn.utils.clip_grad_norm_(params, 1.0)`
- [ ] **Phase 1 training loop** (frozen backbone, epochs 1–25):
  - [ ] Backbone in `.eval()`, `requires_grad=False`
  - [ ] Only classifier parameters in optimizer
  - [ ] Batch size = 8
- [ ] **Phase 2 training loop** (unfrozen, epochs 26–50):
  - [ ] Unfreeze last 4 blocks of backbone
  - [ ] Dual learning rates (head=3e-5, backbone=3e-6)
  - [ ] Batch size = 4, gradient accumulation = 2
- [ ] **Phased loss switching**: calls `get_loss_fn(epoch)` each epoch
- [ ] **Batch-level augmentation**: CutMix (25%) + Copy-Paste (30%)
- [ ] **Early stopping**: monitor val mIoU, patience=12
- [ ] **Checkpointing**: save best by val mIoU + every 10 epochs
- [ ] **Logging**: print per-epoch metrics (loss, mIoU, dice, pixel_acc)
- [ ] **History saving**: loss curves, metric curves to `train_stats/`
- [ ] Verify: training runs for at least 3 epochs without crash on Kaggle

---

### MEMBER 2 — Dataset Tools + Report Prep

#### 2A. Dataset Audit (`audit_dataset.py`)
- [ ] Count images per split (train/val/test)
- [ ] Compute pixel-level class distribution from ALL train masks (not sampled):
  - [ ] Per-class pixel count
  - [ ] Per-class pixel percentage
  - [ ] Per-class image count (how many images contain this class)
- [ ] **Same analysis for val and test sets**
- [ ] Output results to `dataset_audit_results.txt`
- [ ] Generate class distribution bar chart → `class_distribution.png`
- [ ] Identify: which classes are in test but not in train? Vice versa?
- [ ] **KEY OUTPUT**: computed class weights (inverse sqrt frequency):
  ```python
  weights = 1.0 / torch.sqrt(pixel_frequencies + 1e-3)
  weights = weights / weights.sum() * num_classes
  ```
- [ ] Save weights to a file that `train.py` can load

#### 2B. Enhanced Dataset Class (`dataset.py`)
- [ ] `MaskDatasetAlbumentations` class:
  - [ ] Reads images with `cv2.imread` (RGB)
  - [ ] Reads masks with `PIL.Image.open` (handles 16-bit)
  - [ ] Converts mask raw values → class IDs using `value_map`
  - [ ] Applies Albumentations transforms to BOTH image and mask simultaneously
  - [ ] Returns `(image_tensor, mask_tensor)` for train, `(image_tensor, mask_tensor, filename)` for test
- [ ] `MaskDatasetVal` class (no augmentation, just resize+normalize)
- [ ] `build_dataloaders(config)` helper function
  - [ ] Returns `train_loader, val_loader`
  - [ ] Training uses `WeightedRandomSampler` (3x weight for images with Logs/Flowers)
  - [ ] `num_workers=2, pin_memory=True`
- [ ] Verify: iterate one batch, check shapes, check mask values are 0–9

#### 2C. Rare Class Tools (`rare_class_tools.py`)
- [ ] `build_rare_class_pool(mask_dir, image_dir, class_ids, min_area=100)`
  - [ ] Scans all training masks
  - [ ] For each target class: find connected components
  - [ ] Extract bounding box crop of image + mask for each component
  - [ ] Store as list of `(image_crop, mask_crop)` tuples
  - [ ] Print: "Found N instances of Logs, M instances of Flowers"
- [ ] `copy_paste_augment(batch_images, batch_masks, pool, p=0.3)`
  - [ ] Randomly selects crops from pool
  - [ ] Pastes at random position in batch images
  - [ ] Updates masks accordingly
  - [ ] Handles edge cases (crop larger than target area)
- [ ] `WeightedRandomSampler` setup for rare class images
- [ ] Verify: visualize 5 copy-paste results — mask alignment correct

#### 2D. Ablation Baseline Training (Kaggle Notebook 2)
- [ ] Copy the ORIGINAL `train_segmentation.py` with ONLY these changes:
  - [ ] SGD → AdamW (`lr=1e-4, weight_decay=0.01`)
  - [ ] Add computed class weights to `CrossEntropyLoss(weight=...)`
  - [ ] Epochs: 10 → 50
  - [ ] Add `CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)`
  - [ ] **NO augmentation, NO Lovász, NO decoder changes**
- [ ] Train on Kaggle
- [ ] Record: final val mIoU, per-class IoU, loss curve
- [ ] Save: `model_baseline.pth`, `baseline_metrics.txt`, `baseline_curves.png`

#### 2E. Report Writing (REPORT.md → REPORT.pdf)
- [ ] **Page 1: Title + Abstract**
  - [ ] Team name
  - [ ] One-paragraph summary of approach and best mIoU
- [ ] **Page 2: Methodology**
  - [ ] Model architecture diagram (DINOv2 + decoder)
  - [ ] Two-phase training strategy description
  - [ ] Loss function design with rationale
- [ ] **Page 3: Data Analysis**
  - [ ] Class distribution histogram
  - [ ] Train vs Test distribution comparison
  - [ ] Augmentation pipeline listing with justification for each
- [ ] **Page 4: Ablation Study Results**
  - [ ] Comparison table (baseline / +optimizer / +augmentation / +Lovász / full)
  - [ ] Per-class IoU bar charts for each configuration
  - [ ] Training loss curves overlay
- [ ] **Page 5: Per-Class Analysis**
  - [ ] Confusion matrix heatmap
  - [ ] Per-class IoU scores with analysis
  - [ ] Side-by-side GT vs prediction images (3-4 good, 2-3 failure cases)
- [ ] **Page 6: Challenges & Solutions**
  - [ ] Challenge 1: Class imbalance → copy-paste + weighted loss
  - [ ] Challenge 2: Domain shift (train→test) → DINOv2 + photometric aug
  - [ ] Challenge 3: Compute constraints → frozen backbone + AMP
  - [ ] Challenge 4: Logs at 0.07% → targeted copy-paste augmentation
- [ ] **Page 7: Failure Cases**
  - [ ] 3-4 images where model fails
  - [ ] Analysis of WHY it fails (class confusion, boundary errors, etc.)
- [ ] **Page 8: Conclusion & Future Work**
  - [ ] Summary of best approach and final metrics
  - [ ] What would improve with more compute
  - [ ] Potential real-world deployment considerations
- [ ] Convert to PDF (max 8 pages)

---

### MEMBER 3 — Inference Pipeline + Packaging

#### 3A. Test Script with TTA (`test.py`)
- [ ] Modify `test_segmentation.py` to support TTA:
  - [ ] `--tta` flag to enable/disable TTA
  - [ ] 4 variants: original, horizontal flip, scale 0.75×, scale 1.25×
  - [ ] Average softmax probabilities → argmax
  - [ ] Handle DINOv2 patch size constraint (input dims must be ×14)
- [ ] Works on val set (with GT masks → computes metrics)
- [ ] Works on test set (with or without GT masks → saves predictions)
- [ ] Saves: raw masks, colored masks, comparison images
- [ ] Saves: `evaluation_metrics.txt` with per-class IoU
- [ ] **Per-class IoU reporting** (not just mean):
  ```
  Background:     0.0000
  Trees:          0.XXXX
  Lush Bushes:    0.XXXX
  Dry Grass:      0.XXXX
  ...
  Mean IoU:       0.XXXX
  ```
- [ ] Supports loading both DINOv2-Small and Base (match whatever was trained)
- [ ] `--model_path` argument works correctly
- [ ] Verify: runs on val set, produces reasonable IoU numbers

#### 3B. TTA Module (`tta.py`)
- [ ] `predict_with_tta(model, backbone, image, device)` function
  - [ ] Input: single image tensor `[1, 3, H, W]`
  - [ ] Output: class prediction mask `[1, H, W]`
  - [ ] Handles scale variants (pad to ×14 multiples)
  - [ ] Handles flip reversal (flip prediction back)
  - [ ] Averages softmax probabilities across 4 variants
- [ ] Verify: TTA prediction ≠ non-TTA prediction (they should differ slightly)
- [ ] Benchmark: TTA adds how many ms per image?

#### 3C. Backbone Unfreeze Experiment (Kaggle Notebook 3)
- [ ] Copy Member 1's code WITH augmentations
- [ ] Change: fully unfreeze entire DINOv2 backbone from epoch 1
- [ ] Change: dual optimizer LRs: head=1e-4, backbone=1e-5
- [ ] Change: batch_size=2 (to fit unfrozen model in VRAM)
- [ ] Epochs: 50
- [ ] Add `torch.cuda.amp` for FP16
- [ ] Train on Kaggle
- [ ] Record: final val mIoU, per-class IoU, loss curve
- [ ] Save: `model_unfrozen.pth`, `unfrozen_metrics.txt`
- [ ] Compare val mIoU with Member 1's model → share results

#### 3D. Visualization Improvements (`visualize.py`)
- [ ] Generate confusion matrix from val predictions
  - [ ] `sklearn.metrics.confusion_matrix`
  - [ ] Plot as heatmap with class names
  - [ ] Save as `confusion_matrix.png`
- [ ] Generate per-class IoU bar chart (already in test script, verify quality)
- [ ] Generate failure case gallery:
  - [ ] Find 5 images with LOWEST IoU
  - [ ] Save side-by-side: Input | GT | Prediction
  - [ ] Annotate: which classes are confused
- [ ] Generate augmentation showcase:
  - [ ] Show 1 original image + 6 augmented versions with masks
  - [ ] For the report — proves augmentation diversity

#### 3E. Packaging & README
- [ ] **requirements.txt**:
  ```
  torch>=2.0.0
  torchvision>=0.15.0
  numpy
  Pillow
  opencv-python
  albumentations
  matplotlib
  tqdm
  pyyaml
  ```
- [ ] **README.md**:
  - [ ] Project title and team info
  - [ ] Environment setup instructions (step by step)
  - [ ] How to install dependencies
  - [ ] How to train:
    ```bash
    python train.py --config config.yaml
    ```
  - [ ] How to run inference:
    ```bash
    python test.py --model_path weights/best_model.pth --data_dir path/to/test --tta
    ```
  - [ ] How to reproduce exact results
  - [ ] Expected outputs
  - [ ] Model architecture summary
  - [ ] Final metrics achieved
- [ ] **Submission folder structure** (verify everything is present):
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
  │   ├── masks/
  │   ├── masks_color/
  │   └── comparisons/
  └── REPORT.pdf
  ```
- [ ] **Verify submission**: can a fresh Kaggle notebook load weights + run `test.py` from scratch?

---

## 🟢 PHASE 2: Training Runs (Days 2–3)

### All Members — Parallel Training

| Notebook | Member | What's Training | Expected Time | Expected mIoU |
|---|---|---|---|---|
| Kaggle 1 | Member 1 | Full pipeline (Phase 1: frozen, 25 epochs) | 2–4 hours | 0.45–0.55 |
| Kaggle 2 | Member 2 | Ablation baseline (AdamW+weights only, 50 epochs) | 2–3 hours | 0.40–0.50 |
| Kaggle 3 | Member 3 | Unfrozen backbone experiment (50 epochs) | 3–5 hours | 0.45–0.60 |

**Checkpoints**: save to `/kaggle/working/` every 5 epochs. Download `.pth` files.

#### After Phase 1 Training (Member 1 only):
- [ ] Check val mIoU — is it > 0.40? If not, debug before Phase 2
- [ ] Check per-class IoU — which classes are near zero?
- [ ] If Logs IoU ≈ 0: activate copy-paste augmentation for Phase 2
- [ ] Start Phase 2 (unfreeze last 4 blocks, continue training 25 more epochs)

#### After ALL Training Runs Complete:
- [ ] **Member 1** reports: final val mIoU = ___
- [ ] **Member 2** reports: final val mIoU = ___
- [ ] **Member 3** reports: final val mIoU = ___
- [ ] **DECISION**: which model becomes the submission?
  - [ ] If Member 1 > others → submit Member 1's model ✅
  - [ ] If Member 3 > Member 1 → retrain with Member 3's approach using Member 1's full pipeline
- [ ] Fill in ablation table with real numbers

---

## 🔵 PHASE 3: Test Predictions + TTA (Day 3–4)

- [ ] Load best model weights
- [ ] Run `test.py --tta` on **validation set** first → record mIoU with TTA
- [ ] Compare: mIoU without TTA vs with TTA → record the improvement
- [ ] Run `test.py --tta` on **test set** → save all prediction masks
- [ ] Verify predictions visually: spot-check 10 random test images
  - [ ] Does Sky look right?
  - [ ] Are Rocks being detected? (17.7% of test data — must be accurate)
  - [ ] Is model hallucinating Flowers/Logs on test data? (they shouldn't be there)
- [ ] If hallucination problem: add post-processing confidence threshold for rare classes
- [ ] Save all outputs to `predictions/` folder

---

## 🟣 PHASE 4: Report + Final Packaging (Day 4–5)

### Member 2 — Report Finalization
- [ ] Insert real numbers into ablation table
- [ ] Insert real training curves from all 3 experiments
- [ ] Insert real confusion matrix from best model
- [ ] Insert real failure case images
- [ ] Proofread all sections
- [ ] Export to PDF ≤ 8 pages
- [ ] File name: `REPORT.pdf`

### Member 3 — Final Packaging
- [ ] All files are in submission folder (see structure above)
- [ ] `requirements.txt` is complete and tested
- [ ] `README.md` is clear enough for someone unfamiliar
- [ ] Model weights are included (`.pth` file)
- [ ] Predictions are included (`predictions/` folder)
- [ ] **Dry run test**: create new Kaggle notebook, install deps, load model, run test.py → works?

### Member 1 — Final Review
- [ ] Review all code for errors
- [ ] Verify config matches what was actually trained
- [ ] Verify model weights match the architecture in code
- [ ] Verify class mapping is consistent across ALL files:
  ```python
  # THIS MUST BE IDENTICAL in train.py, test.py, dataset.py:
  value_map = {
      0: 0,        # Background
      100: 1,      # Trees
      200: 2,      # Lush Bushes
      300: 3,      # Dry Grass
      500: 4,      # Dry Bushes
      550: 5,      # Ground Clutter
      600: 6,      # Flowers   ← NOT in hackathon class list but in data
      700: 7,      # Logs      ← was index 6 in provided script!
      800: 8,      # Rocks     ← was index 7 in provided script!
      7100: 9,     # Landscape ← was index 8 in provided script!
      10000: 10,   # Sky       ← was index 9 in provided script!
  }
  ```
  ⚠️ **WAIT — the provided script has Flowers as class 600 MISSING from the hackathon class list, but present in data! The script maps 10 classes (0-9) with value_map including 0→0 (Background). The hackathon lists Flowers (600) but the script doesn't have it. VERIFY the actual class mapping before training! Use the provided script's mapping unless you find 600 in masks.**

---

## 🔶 CRITICAL VERIFICATION GATES

### Gate 1: Before ANY Training
- [ ] Augmented image + mask pair visually verified (mask aligns with image after augmentation)
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
- [ ] test.py runs on test set and produces predictions for ALL 1,002 images
- [ ] Predictions folder has 1,002 mask files
- [ ] Report PDF is ≤ 8 pages
- [ ] README has working commands
- [ ] Class mapping is consistent across ALL Python files
- [ ] No test images were used in training (instant DQ if violated)

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

| Time | What |
|---|---|
| **Day 1 End** | All 3 members confirm: environment works, code compiles, can run 1 training epoch |
| **Day 2 End** | Member 1: Phase 1 mIoU = ___. Member 2: Baseline mIoU = ___. Member 3: Unfrozen training started. |
| **Day 3 End** | All training done. Best model selected. Ablation table filled. TTA tested. |
| **Day 4 End** | Test predictions generated. Report 80% done. README drafted. |
| **Day 5 End** | Everything packaged. Dry-run verified. SUBMIT. |

---

## 🚨 EMERGENCY FALLBACKS

| Problem | Fallback |
|---|---|
| DINOv2-Base OOMs | Use DINOv2-Small (already in provided script) |
| Lovász causes NaN | Use `WeightedCE + Dice` only (skip Lovász entirely) |
| Backbone unfreeze OOMs | Keep backbone 100% frozen. Focus on decoder + augmentation. |
| Augmentation corrupts masks | Disable Albumentations. Use only `HorizontalFlip` via torchvision. |
| Val mIoU stuck < 0.35 | Check class mapping. Check mask normalization (`*255` in original script is suspicious). Check that optimizer is AdamW not SGD. |
| Kaggle session dies | Always checkpoint. Build `--resume` flag into train.py. |
| TTA dimension mismatch | Skip scale variants. Use only original + horizontal flip (2 variants). |
| Report too long | Cut Failure Cases to 1 page. Cut Future Work to 3 sentences. |
| Can't reproduce results | Lock `seed=42` in torch, numpy, random, CUDA. Set `deterministic=True`. |

---

> **🏁 "The model that wins isn't the most complex. It's the one trained with the right data, the right loss, on the right hardware, submitted on time."**
