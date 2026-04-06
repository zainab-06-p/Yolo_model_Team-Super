# 🧠 Offroad Semantic Segmentation — Model Architecture & Training Blueprint
> Duality AI Hackathon | Desert Environment Pixel-Level Classification

---

## 📌 Table of Contents
1. [Core Philosophy](#core-philosophy)
2. [Architecture Overview](#architecture-overview)
3. [Multi-Model Federated Ensemble System](#multi-model-federated-ensemble-system)
4. [Dataset Processing Pipeline](#dataset-processing-pipeline)
5. [Training Strategy](#training-strategy)
6. [Generalization & Domain Shift Handling](#generalization--domain-shift-handling)
7. [Pixel-Level Classification Design](#pixel-level-classification-design)
8. [Full Feature Checklist](#full-feature-checklist)
9. [Inference & Testing Pipeline](#inference--testing-pipeline)
10. [Tech Stack](#tech-stack)

---

## 1. Core Philosophy

The model must:
- **See every pixel** — not just bounding boxes or regions
- **Generalize to unseen desert locations** — not overfit to training coordinates
- **Be fast enough** — target < 50ms per image at inference
- **Be accurate across all 10 classes** — including rare ones like Logs and Flowers

The guiding principle: **train on patterns, not places.**

---

## 2. Architecture Overview

### Primary Backbone: SegFormer (MiT-B2 or MiT-B3)

**Why SegFormer?**
- Transformer-based encoder → captures global context (sky, horizon, landscape relationships)
- Hierarchical patch merging → multi-scale feature extraction without heavy computation
- Lightweight MLP decoder → fast at inference, avoids the slowness of FPN/ASPP
- State-of-the-art IoU on outdoor segmentation benchmarks
- No positional encoding → better generalization to unseen spatial layouts

```
Input Image (H x W x 3)
        ↓
[ SegFormer Encoder: MiT-B2 ]
  Stage 1: 1/4 resolution  → fine texture features
  Stage 2: 1/8 resolution  → local structure features
  Stage 3: 1/16 resolution → semantic features
  Stage 4: 1/32 resolution → global context features
        ↓
[ Multi-Scale Feature Fusion (MLP Decoder) ]
  All stages upsampled → concatenated → fused
        ↓
[ Classification Head ]
  1x1 Conv → 10-class Softmax
        ↓
Output Segmentation Map (H x W x 10)
```

### Secondary / Ensemble Model: DeepLabV3+ (ResNet-50 or EfficientNet-B4 backbone)

**Why DeepLabV3+?**
- ASPP (Atrous Spatial Pyramid Pooling) → captures multi-scale context in parallel
- Excellent for fine boundary detection (rocks vs ground)
- Encoder-decoder structure retains spatial detail
- CNN-based → fast at inference, complementary to SegFormer's transformer view

### Optional Lightweight Model: DDRNet-23-slim

**Why DDRNet?**
- Real-time dual-resolution network
- Keeps both high-res (spatial detail) and low-res (semantic context) paths
- Used as the speed-focused model in the ensemble
- < 20ms inference on GPU

---

## 3. Multi-Model Federated Ensemble System

### Architecture: Weighted Soft-Voting Ensemble

The ensemble combines predictions from 2–3 models **at the logit/probability level**, not the label level. This preserves uncertainty information.

```
Image
  ├──→ SegFormer (MiT-B2)    → Probability Map P1 [H×W×10]
  ├──→ DeepLabV3+ (ResNet50) → Probability Map P2 [H×W×10]
  └──→ DDRNet-23-slim        → Probability Map P3 [H×W×10]  (optional, for speed)
              ↓
  Weighted Average:
  P_final = w1*P1 + w2*P2 + w3*P3
  (w1=0.5, w2=0.35, w3=0.15 — tune via validation IoU)
              ↓
  argmax(P_final) → Final Segmentation Map
```

### Why This Is Fast Enough
- Each model runs independently (parallelizable on multi-GPU or sequentially on single GPU)
- No cross-model attention or communication overhead
- Soft voting is just a weighted average — negligible extra cost (~0.1ms)
- Total inference time target: < 45ms (within 50ms limit)

### Ensemble Weight Tuning
- Run each model separately on the validation set
- Record per-class and overall IoU
- Assign higher weights to better-performing models
- Use Optuna or grid search to tune weights

---

## 4. Dataset Processing Pipeline

### Step 1: Dataset Audit
```
- Count images per class
- Detect class imbalance (expected: Logs, Flowers will be rare)
- Visualize class distribution histogram
- Check image resolution consistency
- Verify annotation quality on a sample
```

### Step 2: Class ID Remapping
Raw IDs (100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000) must be mapped to contiguous indices (0–9) for the loss function.

```python
CLASS_MAP = {
    100:   0,   # Trees
    200:   1,   # Lush Bushes
    300:   2,   # Dry Grass
    500:   3,   # Dry Bushes
    550:   4,   # Ground Clutter
    600:   5,   # Flowers
    700:   6,   # Logs
    800:   7,   # Rocks
    7100:  8,   # Landscape
    10000: 9,   # Sky
}
```

### Step 3: Data Augmentation Pipeline

Applied **only to training set**, not validation or test.

#### Geometric Augmentations (always safe for segmentation)
- [ ] Random horizontal flip (p=0.5)
- [ ] Random vertical flip (p=0.2)
- [ ] Random rotation (±15°, with border reflection)
- [ ] Random crop and resize (scale 0.5–2.0, crop to training size)
- [ ] Random perspective distortion (p=0.3)
- [ ] Elastic transform (for texture shape variation)
- [ ] Grid distortion (simulates terrain undulation)
- [ ] Coarse dropout / cutout (simulate occlusion — critical for Logs class)
- [ ] Random resized crop (standard for training)

#### Photometric Augmentations (simulate lighting/sensor variation)
- [ ] Color jitter (brightness ±0.3, contrast ±0.3, saturation ±0.2, hue ±0.05)
- [ ] Gaussian blur (p=0.2, kernel 3–7)
- [ ] Random grayscale (p=0.1 — forces model to not rely on color alone)
- [ ] Gaussian noise injection
- [ ] Random gamma correction (simulate different times of day)
- [ ] CLAHE (Contrast Limited Adaptive Histogram Equalization) — p=0.3
- [ ] Random brightness/shadow simulation (simulate cloud cover)
- [ ] Motion blur (p=0.1 — simulate UGV movement)
- [ ] Sun flare simulation (p=0.1)
- [ ] Fog/haze simulation (p=0.1 — domain robustness)

#### Synthetic Augmentation (domain shift robustness)
- [ ] MixUp between training samples (blends two images and their masks)
- [ ] CutMix (cuts a patch from one image and pastes into another with its mask)
- [ ] Copy-paste augmentation (paste rare class instances — Logs, Flowers — into other images)

### Step 4: Normalization
```python
mean = [0.485, 0.456, 0.406]  # ImageNet mean (for pretrained backbones)
std  = [0.229, 0.224, 0.225]  # ImageNet std
```

### Step 5: DataLoader Configuration
```
- Training batch size: 8–16 (depending on GPU VRAM)
- Validation batch size: 4
- num_workers: 4–8
- pin_memory: True
- Training image size: 512×512 or 640×640
- Shuffle: True (training only)
```

---

## 5. Training Strategy

### Loss Function: Composite Loss

Using a combination of losses is critical for handling class imbalance and pixel-level precision.

```
Total Loss = α * CE_Loss + β * Dice_Loss + γ * Focal_Loss

Where:
  α = 0.4  (Cross Entropy — baseline classification)
  β = 0.4  (Dice — handles class imbalance, IoU-like optimization)
  γ = 0.2  (Focal — focuses on hard/rare pixels)
```

- **Cross Entropy Loss** with class weights (inverse frequency weighting for rare classes)
- **Dice Loss** — directly optimizes overlap (correlated with IoU metric)
- **Focal Loss** (γ=2) — downweights easy pixels, emphasizes hard-to-classify regions
- **Lovász-Softmax Loss** (optional) — directly optimizes IoU, consider as replacement/addition

### Class Weighting for CE Loss
```python
# Compute from training set annotation frequency
weight[i] = 1 / (freq[i] + epsilon)
# Normalize so weights sum to num_classes
# Rare classes (Logs, Flowers) will get much higher weight
```

### Optimizer
```
Optimizer: AdamW
  lr: 6e-5 (encoder), 6e-4 (decoder) — use layer-wise LR
  weight_decay: 0.01
  betas: (0.9, 0.999)
```

### Learning Rate Scheduler
```
Warmup: Linear warmup for first 5% of total steps
Main: Polynomial decay (power=0.9) — standard for segmentation
Alternative: OneCycleLR or CosineAnnealingWarmRestarts
```

### Training Configuration
```
Epochs: 100–150
Early Stopping: patience=20 epochs on validation mIoU
Gradient Clipping: max_norm=1.0
Mixed Precision: FP16 (torch.cuda.amp) — 2x speedup, less VRAM
```

### Fine-Tuning Strategy
1. **Phase 1** — Freeze encoder, train decoder only (5–10 epochs) — warm up new head
2. **Phase 2** — Unfreeze all layers, train end-to-end with low LR (main training)
3. **Phase 3** — Fine-tune with test-time augmentation tuning on validation set

### Checkpointing
- Save best model by **validation mIoU** (not loss)
- Save every 10 epochs as safety backup
- Keep top-3 checkpoints

---

## 6. Generalization & Domain Shift Handling

This is the most critical section — the test set is a **different desert location**.

### Techniques to Prevent Overfitting to Training Location

- [ ] **Strong augmentation** — makes the model learn appearance invariance
- [ ] **DropPath / Stochastic Depth** — regularization in transformer blocks
- [ ] **Label smoothing** (ε=0.1) — prevents overconfident predictions
- [ ] **Test-Time Augmentation (TTA)** — average predictions over flipped/scaled versions
- [ ] **Feature normalization** — Instance Norm or Layer Norm (not Batch Norm) — more stable across domains
- [ ] **Style Transfer Augmentation** (AdaIN) — randomize the "look" of images during training
- [ ] **Domain Randomization** — randomize color, lighting, texture during training
- [ ] **Spectral Normalization** on discriminator if using adversarial training
- [ ] **Self-Supervised Pre-training** on all available images (train + test unlabeled) — MAE/DINO
- [ ] **Pseudo-labeling on test images** — generate pseudo labels with confident predictions, retrain

### Test-Time Augmentation (TTA) Protocol
```
For each test image, run inference on:
  1. Original image
  2. Horizontally flipped
  3. Vertically flipped
  4. Both flipped
  5. Scale 0.75x
  6. Scale 1.25x

Average all 6 probability maps → final prediction
Cost: ~6x inference time (still < 300ms total, acceptable)
```

---

## 7. Pixel-Level Classification Design

### Multi-Scale Context Aggregation
- Model must understand **local** (individual pixel texture) AND **global** (sky is always above ground) context
- SegFormer handles this via hierarchical stages
- ASPP in DeepLabV3+ captures context at scales: 1, 6, 12, 18 dilation

### Boundary Refinement
- [ ] Add boundary loss term — penalize errors near class edges
- [ ] Use CRF (Conditional Random Field) post-processing — refines jagged boundaries
- [ ] DenseCRF as post-processing step (no retraining needed)

### Attention Mechanisms
- [ ] Self-Attention (from SegFormer transformer blocks)
- [ ] Channel Attention (SE blocks in CNN branch)
- [ ] Spatial Attention (CBAM — Convolutional Block Attention Module)
- [ ] Cross-Attention between encoder features of different scales

### Class Confusion Mitigation
Likely confusion pairs based on visual similarity:
```
- Dry Grass ↔ Dry Bushes   (similar color, different structure)
- Rocks ↔ Ground Clutter    (both on ground, similar texture)
- Lush Bushes ↔ Trees       (both green, different scale)
- Logs ↔ Rocks              (both brown/dark, similar shape)
```

Solutions:
- [ ] Confusion matrix analysis during validation
- [ ] Class-specific augmentation for confused pairs
- [ ] Auxiliary classification head on confusing classes

---

## 8. Full Feature Checklist ✅

### Model Architecture
- [ ] Transformer backbone (SegFormer MiT-B2/B3) as primary model
- [ ] CNN backbone (DeepLabV3+ or ResNet) as secondary model
- [ ] Lightweight real-time model (DDRNet) optional for speed
- [ ] Multi-scale feature extraction at 4 hierarchy levels
- [ ] MLP decoder (SegFormer style) — faster than FPN
- [ ] ASPP module for multi-scale context (DeepLabV3+)
- [ ] Skip connections for spatial detail preservation
- [ ] Attention mechanisms (self, channel, spatial)
- [ ] Separate encoder and decoder learning rates
- [ ] Pretrained ImageNet weights for backbone initialization

### Dataset Processing
- [ ] Class ID remapping to 0–9
- [ ] Class frequency analysis and imbalance detection
- [ ] Training/Validation split verification
- [ ] Mean and std computation from training set
- [ ] Image resolution standardization (512×512 or 640×640)
- [ ] Data pipeline profiling (no CPU bottleneck)

### Data Augmentation
- [ ] Random horizontal & vertical flip
- [ ] Random rotation with border handling
- [ ] Random crop with resize
- [ ] Color jitter (brightness, contrast, saturation, hue)
- [ ] Gaussian blur and noise
- [ ] Random grayscale
- [ ] Gamma correction
- [ ] Coarse dropout / cutout (occlusion simulation)
- [ ] MixUp augmentation
- [ ] CutMix augmentation
- [ ] Copy-paste for rare classes (Logs, Flowers)
- [ ] Elastic and grid distortion
- [ ] Shadow and sun flare simulation
- [ ] Motion blur
- [ ] Fog/haze simulation

### Loss Function
- [ ] Cross Entropy with class weights
- [ ] Dice Loss
- [ ] Focal Loss (γ=2)
- [ ] Lovász-Softmax (optional — directly optimizes IoU)
- [ ] Boundary Loss (optional — sharpens edges)
- [ ] Label smoothing (ε=0.1)

### Optimizer & Scheduler
- [ ] AdamW optimizer with weight decay
- [ ] Layer-wise learning rates (encoder < decoder)
- [ ] Linear warmup
- [ ] Polynomial LR decay or CosineAnnealing
- [ ] Gradient clipping (max_norm=1.0)
- [ ] Mixed precision training (FP16)

### Training Best Practices
- [ ] Freeze encoder → train decoder → unfreeze all (3-phase training)
- [ ] Early stopping on validation mIoU
- [ ] Best checkpoint saved by mIoU (not loss)
- [ ] Gradient accumulation if batch size too small
- [ ] SyncBatchNorm if multi-GPU
- [ ] Reproducible seed setting (torch, numpy, random)

### Generalization
- [ ] Strong augmentation pipeline
- [ ] Test-Time Augmentation (TTA)
- [ ] Label smoothing
- [ ] DropPath / stochastic depth in transformer
- [ ] Instance/Layer normalization preference over BatchNorm
- [ ] Pseudo-labeling on test images (optional)
- [ ] Feature consistency regularization

### Evaluation
- [ ] mIoU (mean Intersection over Union) — primary metric
- [ ] Per-class IoU — identify weak classes
- [ ] Pixel Accuracy
- [ ] Confusion Matrix
- [ ] Loss curves (training vs validation)
- [ ] Inference speed benchmark (ms/image)
- [ ] Failure case visualization
- [ ] GradCAM or attention map visualization

### Post-Processing
- [ ] DenseCRF for boundary refinement
- [ ] Connected components filtering (remove tiny noise regions)
- [ ] Hole filling (fill small unlabeled holes within a class)

### Ensemble
- [ ] Soft-voting ensemble across 2–3 models
- [ ] Per-class weight tuning
- [ ] Ensemble validation on val set before submitting
- [ ] TTA applied per model before ensembling

### Code Quality & Submission
- [ ] Modular codebase (config.py, dataset.py, model.py, train.py, test.py, utils.py)
- [ ] Config file (YAML or argparse) for all hyperparameters
- [ ] Logging with wandb or TensorBoard
- [ ] README with full reproduction instructions
- [ ] Model weights saved in standard format (.pth)
- [ ] Visualization script for colored segmentation output
- [ ] Requirements.txt or environment.yml

---

## 9. Inference & Testing Pipeline

```
Test Image
    ↓
Resize to training resolution (512×512)
    ↓
Normalize (ImageNet mean/std)
    ↓
[ TTA: run 6 augmented versions ]
    ↓
Model(s) Forward Pass
    ↓
Average probability maps
    ↓
argmax → raw prediction mask
    ↓
[ Optional: DenseCRF post-processing ]
    ↓
Remap 0–9 back to original class IDs
    ↓
Save colorized mask + raw prediction
    ↓
Compute IoU against ground truth (if available)
```

---

## 10. Tech Stack

| Component | Tool |
|---|---|
| Primary Framework | PyTorch |
| Segmentation Library | `mmsegmentation` or `segmentation_models_pytorch` |
| Augmentation | Albumentations |
| Experiment Tracking | Weights & Biases (wandb) or TensorBoard |
| Mixed Precision | `torch.cuda.amp` |
| Post-processing | `pydensecrf` (DenseCRF) |
| Visualization | Matplotlib, OpenCV |
| Config Management | YAML + argparse |
| Environment | Conda (EDU env from Duality's setup) |

---

## 🏆 Expected Performance Targets

| Metric | Baseline (starter) | Our Target |
|---|---|---|
| mIoU | ~0.31 (given) | > 0.65 |
| Pixel Accuracy | ~70% | > 88% |
| Inference Speed | — | < 50ms |
| Training Time | — | 3–6 hrs (GPU) |

---

> **Next Steps:** Build `dataset.py` → `model.py` → `train.py` → `test.py` → ensemble → submit.
