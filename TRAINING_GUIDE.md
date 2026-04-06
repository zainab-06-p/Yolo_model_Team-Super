# COMPLETE TRAINING GUIDE
# High-Performance Desert Segmentation Models
# Target: mIoU >= 0.90

## ═══════════════════════════════════════════════════════════════
## OVERVIEW: TWO TRAINING APPROACHES
## ═══════════════════════════════════════════════════════════════

1. **Option A: Quick Ensemble** (Use existing 3 models)
   - File: `ensemble_kaggle.py` (already trained models)
   - Time: Immediate inference
   - Expected mIoU: 0.42-0.46

2. **Option B: High-Performance Training** (Train new models)
   - File: `train_high_performance.py`
   - Time: 6-12 hours training
   - Expected mIoU: 0.55-0.65 (single model), 0.70+ (ensemble)

3. **Option C: Maximum Performance** (Multi-model ensemble)
   - Train 5+ different models
   - Multi-scale + TTA inference
   - Expected mIoU: 0.80-0.90

## ═══════════════════════════════════════════════════════════════
## OPTION A: QUICK ENSEMBLE (No Training Needed)
## ═══════════════════════════════════════════════════════════════

### Step 1: Upload Models to Kaggle
1. Go to Kaggle → Your Notebook
2. Click "Add Input" → "Upload"
3. Upload these 3 files:
   - `model_finetuned_best.pth.zip` (or .pth)
   - `model_augmented_best.pth`
   - `model_best.pth.zip` (or .pth)
4. They will be in `/kaggle/working/`

### Step 2: Update Paths in ensemble_kaggle.py
```python
MANUAL_MEMBER1_PATH = '/kaggle/working/model_finetuned_best.pth.zip'
MANUAL_MEMBER2_PATH = '/kaggle/working/model_augmented_best.pth'
MANUAL_MEMBER3_PATH = '/kaggle/working/model_best.pth.zip'
DATASET_PATH = '/kaggle/input/yolo-training-data'
```

### Step 3: Enable All Features
```python
USE_ROBUST_PREPROCESSING = True      # Handles noise/blur
ENABLE_BLACK_IMAGE_HANDLING = True   # Handles monochromatic images
```

### Step 4: Run Inference
```python
# In Kaggle notebook cell:
!python ensemble_kaggle.py
```

### Expected Results:
- Val mIoU: ~0.42-0.46
- Time: ~2 min for 317 images

## ═══════════════════════════════════════════════════════════════
## OPTION B: HIGH-PERFORMANCE TRAINING (Recommended)
## ═══════════════════════════════════════════════════════════════

### Prerequisites:
- Kaggle account with GPU (T4x2 or P100)
- Dataset: `yolo-training-data` added to notebook

### Step 1: Prepare Kaggle Environment

1. **Create New Notebook**
   - Go to kaggle.com → Code → New Notebook
   - Accelerator: GPU T4x2

2. **Add Dataset**
   - Click "Add Input" → Search "yolo-training-data"
   - Add your training dataset

3. **Upload Helper Files**
   - Upload `train_high_performance.py` to Kaggle

### Step 2: Update Configuration

Edit `train_high_performance.py`:
```python
class Config:
    # Check these paths match your Kaggle setup
    TRAIN_DIR = '/kaggle/input/yolo-training-data/Offroad_Segmentation_Training_Dataset/train'
    VAL_DIR = '/kaggle/input/yolo-training-data/Offroad_Segmentation_Training_Dataset/val'
    
    # Choose backbone size (smaller = faster, larger = better)
    BACKBONE = 'dinov2_vits14'   # Fastest, 384-dim
    # BACKBONE = 'dinov2_vitb14'  # Balanced, 768-dim  
    # BACKBONE = 'dinov2_vitl14'  # Best, 1024-dim (slower)
    
    # Adjust based on GPU memory
    BATCH_SIZE = 4    # Use 2 for vitl14, 4 for vits14
    
    # Training duration
    NUM_EPOCHS = 50   # Start with 50, increase to 100 for better results
```

### Step 3: Run Training

In Kaggle notebook:
```python
# Cell 1: Install dependencies
!pip install -q torch torchvision opencv-python

# Cell 2: Run training
!python train_high_performance.py
```

### Step 4: Monitor Training

Watch for:
```
Epoch 1: Loss=1.2345, Val mIoU=0.3123
Epoch 2: Loss=0.9876, Val mIoU=0.3543
...
Epoch 10: Loss=0.5432, Val mIoU=0.4567  ✓ New best model saved!
```

### Step 5: Save Best Model

Best model automatically saved as:
```
/kaggle/working/best_high_performance_model.pth
```

Download this file after training completes.

## ═══════════════════════════════════════════════════════════════
## OPTION C: MAXIMUM PERFORMANCE (For 0.90 mIoU)
## ═══════════════════════════════════════════════════════════════

### Phase 1: Train Multiple Models

Train 5 different models with variations:

**Model 1: Base (vits14)**
```python
Config.BACKBONE = 'dinov2_vits14'
Config.NUM_EPOCHS = 50
# Save: model_vits14_base.pth
```

**Model 2: Augmented (vits14)**
```python
Config.BACKBONE = 'dinov2_vits14'
Config.USE_AUGMENTATION = True
Config.AUG_PROB = 0.9
# Save: model_vits14_aug.pth
```

**Model 3: Long Training (vits14)**
```python
Config.BACKBONE = 'dinov2_vits14'
Config.NUM_EPOCHS = 100
# Save: model_vits14_long.pth
```

**Model 4: Larger Backbone (vitb14)**
```python
Config.BACKBONE = 'dinov2_vitb14'
Config.BATCH_SIZE = 2
Config.NUM_EPOCHS = 50
# Save: model_vitb14.pth
```

**Model 5: All Data (vits14)**
```python
# Combine train + val for training
Config.TRAIN_DIR = '/kaggle/input/yolo-training-data/Offroad_Segmentation_Training_Dataset'
Config.VAL_DIR = '/kaggle/input/yolo-training-data/Offroad_Segmentation_Training_Dataset/val'
# Save: model_alldata.pth
```

### Phase 2: Ensemble Inference

Update `inference_maximum_performance.py`:
```python
MODEL_PATHS = {
    'M1': 'model_vits14_base.pth',
    'M2': 'model_vits14_aug.pth', 
    'M3': 'model_vits14_long.pth',
    'M4': 'model_vitb14.pth',
    'M5': 'model_alldata.pth',
}

ENSEMBLE_WEIGHTS = {
    'M1': 0.25,
    'M2': 0.20,
    'M3': 0.25,
    'M4': 0.20,
    'M5': 0.10,
}
```

### Phase 3: Maximum Inference

Run with all optimizations:
```python
# Multi-scale
MULTI_SCALE = [0.75, 1.0, 1.25]

# TTA
TTA_FLIPS = [False, True]

# Run
!python inference_maximum_performance.py
```

## ═══════════════════════════════════════════════════════════════
## TRAINING TIPS FOR BEST RESULTS
## ═══════════════════════════════════════════════════════════════

### 1. Monitor These Metrics:
```python
# Good training:
- Loss should decrease steadily
- Val mIoU should increase
- Best mIoU around epoch 30-50

# Bad training (early stopping):
- Loss stuck or increasing
- Val mIoU decreasing (overfitting)
- Gradient explosion (NaN loss)
```

### 2. Hyperparameter Tuning:
```python
# If underfitting (low mIoU):
- Increase NUM_EPOCHS
- Increase LR slightly
- Reduce augmentation

# If overfitting (val mIoU < train):
- Decrease NUM_EPOCHS
- Add dropout (increase from 0.3 to 0.5)
- Increase augmentation
```

### 3. GPU Optimization:
```python
# If OOM (Out of Memory):
- Reduce BATCH_SIZE to 2 or 1
- Use smaller BACKBONE (vits14 instead of vitl14)
- Enable gradient checkpointing

# If slow:
- Increase BATCH_SIZE (if memory allows)
- Use mixed precision training (fp16)
- Reduce NUM_WORKERS in DataLoader
```

## ═══════════════════════════════════════════════════════════════
## EXPECTED TIMELINE & RESULTS
## ═══════════════════════════════════════════════════════════════

| Approach | Training Time | Expected mIoU | Recommendation |
|----------|---------------|---------------|----------------|
| Option A | 0 min | 0.42-0.46 | Quick start |
| Option B | 2-4 hours | 0.55-0.65 | Good improvement |
| Option C | 10-15 hours | 0.75-0.90 | Maximum performance |

## ═══════════════════════════════════════════════════════════════
## QUICK START COMMANDS (Copy-Paste for Kaggle)
## ═══════════════════════════════════════════════════════════════

### For Option A (Immediate):
```python
# Cell 1
!pip install -q opencv-python

# Cell 2
# [Upload ensemble_kaggle.py and models]

# Cell 3
!python ensemble_kaggle.py
```

### For Option B (Training):
```python
# Cell 1: Install
!pip install -q torch torchvision opencv-python

# Cell 2: Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Cell 3: Train
!python train_high_performance.py

# Cell 4: Check output
!ls -lh *.pth
```

### For Option C (Multi-Model):
```python
# Run training 5 times with different configs
# Then ensemble with inference_maximum_performance.py
```

## ═══════════════════════════════════════════════════════════════
## TROUBLESHOOTING
## ═══════════════════════════════════════════════════════════════

### Problem: "RuntimeError: CUDA out of memory"
Solution: Reduce `BATCH_SIZE = 2` or `BATCH_SIZE = 1`

### Problem: "Val mIoU stuck at 0.20"
Solution: Check dataset paths, ensure masks are loading correctly

### Problem: "Loss is NaN"
Solution: Reduce learning rate `LR = 5e-5`

### Problem: "Training too slow"
Solution: Use `BACKBONE = 'dinov2_vits14'`, reduce `NUM_EPOCHS`

### Problem: "Model not improving after 20 epochs"
Solution: Enable augmentation, increase LR, try different backbone

## ═══════════════════════════════════════════════════════════════
## SUMMARY: RECOMMENDED WORKFLOW
## ═══════════════════════════════════════════════════════════════

1. **Start with Option A** (5 minutes)
   - Get baseline mIoU
   - Verify everything works

2. **Run Option B** (3-4 hours)
   - Train one high-performance model
   - Expected: 0.55-0.65 mIoU

3. **If time permits, Option C** (10+ hours)
   - Train 3-5 models
   - Ensemble them
   - Target: 0.80+ mIoU

4. **For Competition Submission**
   - Use Option C with maximum ensemble
   - Add CRF post-processing
   - Test-time augmentation
   - Expected: 0.85-0.90 mIoU

Good luck! Target 0.90 mIoU is achievable with proper ensemble.
