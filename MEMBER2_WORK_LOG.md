# Member 2 - Work Documentation Log
## Duality AI Hackathon - Semantic Segmentation

---

## 📋 What We Did Today

### 1. Dataset Preparation & Upload (2 hours)
**The Problem:**
- Had a 2.8GB local dataset that needed to go on Kaggle for training
- Browser upload kept failing, API upload gave 401 errors
- Person 1's Google Drive upload was taking 3+ hours

**What We Did:**
- Compressed dataset using 7-Zip (got it down to 2.66GB from 2.8GB)
- Created proper folder structure for Kaggle
- Tried Kaggle API multiple times but token kept failing
- Eventually uploaded via Kaggle browser interface
- Dataset name: `yolo-training-data`
- Final path: `/kaggle/input/datasets/adiinamdar/yolo-training-data/`

**Files in Dataset:**
- Train: 5,714 files (2.33 GB) - Color_Images + Segmentation masks
- Val: 634 files (276 MB)
- Total: 6,348 files

---

### 2. Dataset Audit (30 min)
**What We Did:**
Ran `audit_dataset.py` locally to understand our data

**Key Findings:**
- 10 classes total
- Extreme class imbalance detected
- **Sky** class dominates (~60% of pixels)
- **Logs** class is extremely rare (~0.07% of pixels) - this is a problem!
- **Ground Clutter** also rare (~0.3% pixels)

**Output Files Created:**
- `class_weights.pt` - Computed inverse frequency weights (CRITICAL for team)
- `class_distribution.png` - Pie chart showing imbalance
- `train_vs_test_distribution.png` - Distribution comparison
- `dataset_audit_results.txt` - Full stats

**Why This Matters:**
Without these weights, the model would ignore rare classes like Logs entirely. Member 1 and 3 NEED this file for their training.

---

### 3. Code Fixes (45 min)
**Issue 1: Wrong Dataset Paths**
- Error: `FileNotFoundError: No such file or directory`
- Root cause: Kaggle path structure was different than expected
- Fix: Updated CONFIG to use actual path with `/datasets/` prefix

**Before:**
```python
'/kaggle/input/yolo-training-data/Offroad_Segmentation_Training_Dataset/train'
```

**After:**
```python
'/kaggle/input/datasets/adiinamdar/yolo-training-data/Offroad_Segmentation_Training_Dataset/train'
```

**Issue 2: Albumentations API Changed**
- Error: `ValidationError: scale must be >= 0 and <= 1`
- Root cause: `RandomResizedCrop` syntax changed in newer version
- Fix: Updated to new API format

**Before:**
```python
A.RandomResizedCrop(
    height=h, width=w,
    scale=(0.5, 1.5),  # Old API allowed >1
    ...
)
```

**After:**
```python
A.RandomResizedCrop(
    size=(h, w),  # Now uses size tuple
    scale=(0.5, 1.0),  # Max must be <= 1.0
    ...
)
```

---

### 4. Training Started (Ongoing)
**Current Status:**
- Training is running on Kaggle with GPU T4 x2
- 50 epochs, batch size 8
- Frozen DINOv2 backbone, only training segmentation head

**What We're Seeing:**
- IoU hovering around **0.35** for all epochs so far
- This is... not great. Target is 0.52+

**Why IoU Might Be Stuck at 0.35:**
1. **Frozen backbone** - We're not fine-tuning DINOv2, just the head
2. **Extreme class imbalance** - Model learns to predict common classes (Sky, Ground) and ignores rare ones
3. **Augmentation might be too aggressive** - Could be distorting features too much
4. **Need more epochs** - Sometimes it takes 20+ epochs to see improvement

**What to Do:**
- Let it run for at least 25-30 epochs
- If still stuck at 0.35, we might need to:
  - Reduce augmentation intensity
  - Increase learning rate
  - Or accept that frozen backbone limits performance (Member 1 is doing fine-tuning)

---

## 🚨 Critical Issues Faced

### Issue 1: Kaggle API Authentication Failed
**Symptom:** 401 Unauthorized error
**Attempts:**
- Tried with username `adiin007` - failed
- Tried with `adiinamdar` - failed
- Added `KGAT_` prefix to token - still failed
- Moved kaggle.json to user home - still failed

**Resolution:** Gave up on API, used browser upload instead

### Issue 2: Dataset Too Big for Direct Upload
**Symptom:** Browser upload would hang or fail
**Fix:** Used 7-Zip compression to reduce size slightly

### Issue 3: Albumentations Version Mismatch
**Symptom:** `RandomResizedCrop` threw validation errors
**Fix:** Updated to new API syntax (size tuple instead of height/width)

---

## 📊 Current Outputs

### Already Generated:
1. ✅ `audit_results/class_weights.pt` - Share with team ASAP
2. ✅ `audit_results/class_distribution.png` - For report
3. ✅ `audit_results/train_vs_test_distribution.png` - For report
4. ✅ Dataset uploaded to Kaggle
5. ✅ Training code fixed and running

### Still Waiting For:
1. ⏳ `model_augmented_best.pth` - Best model checkpoint
2. ⏳ `training_curves_aug.png` - Loss/IoU plots
3. ⏳ Final mIoU number - Need this for ablation table

---

## 🎯 Next Steps (While Training Runs)

### Immediate (Do Now):
1. **Share class_weights.pt with Member 1 & 3** - They're waiting for this
2. **Write Report Section 3** - Data Analysis (use the audit charts we have)
3. **Write Report Section 6** - Challenges faced (document the issues above)

### After Training Completes (~2 more hours):
1. Download `model_augmented_best.pth`
2. Record final mIoU number
3. Download training curves plot
4. Compare with Member 1's results for ablation study

### If IoU Stays at 0.35:
- Don't panic - this is the "augmentation" experiment
- Member 1 (fine-tuning) should get better results
- Member 3 (hyperparameter tuning) might help
- Our job was to prove augmentation + class weights work
- Even 0.35 is acceptable if we can show improvement over baseline

---

## 📞 Team Coordination Notes

**Shared with Team:**
- [ ] class_weights.pt
- [ ] class_distribution.png
- [ ] Final mIoU number (when ready)

**Need from Team:**
- Member 1's fine-tuning results for comparison
- Member 3's hyperparameter results

**Blockers for Others:**
- None - class_weights.pt is the only dependency, and it's ready

---

## 💡 Lessons Learned

1. **Kaggle API is flaky** - Browser upload is more reliable for big datasets
2. **Always check actual folder structure** - Kaggle paths have `/datasets/` prefix we didn't expect
3. **Albumentations API changes frequently** - Need to check docs when errors occur
4. **Compress early** - Would have saved time if we compressed from the start
5. **Class imbalance is REAL** - 0.07% for Logs means we need serious augmentation strategies

---

## 🕐 Time Spent So Far

| Task | Time |
|------|------|
| Dataset compression | 20 min |
| Kaggle API attempts | 40 min |
| Browser upload | 30 min |
| Local audit | 15 min |
| Code debugging (paths + albumentations) | 45 min |
| Training (completed) | 2+ hours |
| **Total** | **~5.5 hours** |

---

*Last updated: April 6, 2026 - Training COMPLETE*
