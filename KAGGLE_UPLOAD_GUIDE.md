# 🚀 MEMBER 2 — KAGGLE UPLOAD & TRAINING GUIDE
## Using Your Local Dataset on Kaggle

---

## 📁 YOUR LOCAL DATASET STRUCTURE

Your dataset is located at:
```
c:\Users\adiin\OneDrive\Desktop\yolo\
├── Offroad_Segmentation_Training_Dataset/
│   └── Offroad_Segmentation_Training_Dataset/
│       ├── train/
│       │   ├── Color_Images/     ← 2,857 images
│       │   └── Segmentation/     ← 2,857 masks
│       └── val/
│           ├── Color_Images/     ← 317 images
│           └── Segmentation/     ← 317 masks
├── Offroad_Segmentation_testImages/
│   └── Offroad_Segmentation_testImages/
│       ├── Color_Images/         ← 1,002 images (unlabeled)
│       └── Segmentation/           ← 1,002 masks (for submission format)
```

**Total**: ~4,200 images (2.2GB estimated)

---

## 🔧 STEP 1: PREPARE DATASET ZIP FILE

### Option A: Quick ZIP (Recommended for first upload)

**Windows PowerShell:**
```powershell
# Navigate to your yolo folder
cd C:\Users\adiin\OneDrive\Desktop\yolo

# Create ZIP of training dataset only (smaller, faster)
Compress-Archive -Path "Offroad_Segmentation_Training_Dataset\*" -DestinationPath "desert-dataset.zip" -Force
```

**If ZIP is still too big (>2GB), split it:**
```powershell
# Upload just train folder first
Compress-Archive -Path "Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train" -DestinationPath "desert-train.zip"

# Then val folder separately
Compress-Archive -Path "Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\val" -DestinationPath "desert-val.zip"
```

---

## 🌐 STEP 2: UPLOAD TO KAGGLE

### Method 1: Kaggle Website (Easiest)

1. **Go to**: https://www.kaggle.com
2. **Sign in** with your account
3. **Click**: `+ New Dataset` button (top right)
4. **Fill in**:
   - **Dataset Title**: `desert-dataset` (or any name)
   - **Description**: `Duality AI Hackathon - Desert Segmentation Dataset`
   - **License**: CC0: Public Domain
5. **Drag & Drop**: Your `desert-dataset.zip` file
6. **Click**: `Create` button
7. **Wait**: for upload to complete (depends on your internet speed)

### Method 2: Kaggle CLI (If installed)

```bash
# Install kaggle if not already
pip install kaggle

# Upload dataset
kaggle datasets create -p ./desert-dataset-folder
```

---

## 📝 STEP 3: CREATE KAGGLE NOTEBOOK

1. **Go to**: https://www.kaggle.com/code
2. **Click**: `+ New Notebook`
3. **Settings** (right sidebar):
   - **Accelerator**: GPU T4 x2
   - **Internet**: ON
   - **Language**: Python

---

## 🔌 STEP 4: ADD DATASET TO NOTEBOOK

1. **In notebook**, click `+ Add Data` (top right)
2. **Click**: `Datasets` tab
3. **Search**: `desert-dataset` (your uploaded dataset name)
4. **Click**: `Add` button
5. **Verify path**: The dataset will be available at:
   ```
   /kaggle/input/desert-dataset/Offroad_Segmentation_Training_Dataset/train
   /kaggle/input/desert-dataset/Offroad_Segmentation_Training_Dataset/val
   ```

---

## 📋 STEP 5: COPY YOUR CODE

1. **Open**: `member2_augmentation.py` in your local editor
2. **Select All** (Ctrl+A)
3. **Copy** (Ctrl+C)
4. **Paste** into Kaggle notebook

**Important**: Make sure line 61-62 in your pasted code shows:
```python
'train_dir' : '/kaggle/input/desert-dataset/Offroad_Segmentation_Training_Dataset/train',
'val_dir'   : '/kaggle/input/desert-dataset/Offroad_Segmentation_Training_Dataset/val',
```

If your dataset name is different, update `'desert-dataset'` to match.

---

## ▶️ STEP 6: RUN TRAINING

### Cell-by-Cell Execution (Recommended for first run):

| Cell | Content | Purpose | Expected Output |
|------|---------|---------|-----------------|
| 1 | GPU Check | Verify GPU available | `Tesla T4` |
| 2 | Install libs | albumentations | `Successfully installed` |
| 3 | Imports | torch, cv2, etc. | No errors |
| 4 | CONFIG | Paths, hyperparams | Paths printed |
| 5 | Class mapping | VALUE_MAP | 10 classes shown |
| 6 | Dataset audit | Compute weights | Class distribution chart |
| 7 | Augmentation | 12 transforms | Pipeline defined ✓ |
| 8 | Dataset | Build loaders | `Train: 2857, Val: 317` |
| 9 | Copy-paste | Build rare pool | `Pool: XXX crops` |
| 10 | Model | Define head | `Model defined ✓` |
| 11 | Backbone | Load DINOv2 | `Backbone frozen ✓` |
| 12 | **Training** | 50 epochs | Loss decreasing |
| 13 | Save results | Curves + model | `model_augmented_best.pth` |

### Run All (After verification):
- **Click**: `Run All` button
- **Estimated time**: 2.5–3 hours for 50 epochs
- **Monitor**: Check loss decreases, Val IoU increases

---

## 💾 STEP 7: DOWNLOAD RESULTS

After training completes:

1. **Click**: `Output` tab (in notebook)
2. **Download**:
   - `model_augmented_best.pth` ← **CRITICAL: Your trained model**
   - `class_distribution.png` ← For report
   - `training_curves_aug.png` ← For report
   - `augmentation_check.png` ← Verify augmentations

3. **Share with team**:
   - Your **best Val IoU** number (e.g., `0.5234`)
   - The `class_weights.pt` file (if they need it)

---

## ⚠️ COMMON ISSUES & FIXES

### Issue 1: Dataset path not found
```
FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/...'
```
**Fix**: Check your dataset name matches in CONFIG. Click `+ Add Data` again to verify path.

### Issue 2: Out of Memory (OOM)
```
RuntimeError: CUDA out of memory
```
**Fix**: Reduce batch_size to 4 in CONFIG:
```python
'batch_size' : 4,  # Instead of 8
```

### Issue 3: Albumentations not installed
```
ModuleNotFoundError: No module named 'albumentations'
```
**Fix**: Add this cell at the top:
```python
!pip install albumentations -q
```

### Issue 4: Training too slow
**Fix**: 
- Check GPU is enabled (Settings → GPU T4 x2)
- Reduce num_workers to 0:
```python
'num_workers': 0,
```

---

## 📊 EXPECTED RESULTS

After 50 epochs, you should see:

```
✅ MEMBER 2 TRAINING COMPLETE
   Strategy   : Augmentation + Class Weights (Frozen backbone)
   Best mIoU  : 0.52xx  ← Report this to team!
   Model file : model_augmented_best.pth
```

**Target ranges**:
- **Good**: Val mIoU > 0.45
- **Great**: Val mIoU > 0.55
- **Excellent**: Val mIoU > 0.60

**If Val mIoU < 0.40**: Check class weights are applied and augmentation is working.

---

## 🔄 ALTERNATIVE: RUN LOCALLY FIRST (Quick Test)

If Kaggle upload is slow, test locally first:

```python
# Change CONFIG to local paths temporarily
CONFIG = {
    'train_dir' : r'C:\Users\adiin\OneDrive\Desktop\yolo\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train',
    'val_dir'   : r'C:\Users\adiin\OneDrive\Desktop\yolo\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\val',
    ...
}
```

Run just **Cell 6 (audit)** first to verify everything works:
```python
class_weights = audit_dataset(CONFIG['train_dir'])
```

If that works, proceed to Kaggle for full training.

---

## ✅ PRE-TRAINING CHECKLIST

Before starting 3-hour training, verify:

- [ ] Dataset uploaded to Kaggle successfully
- [ ] Notebook has GPU T4 x2 enabled
- [ ] Added dataset to notebook via `+ Add Data`
- [ ] Paths in CONFIG match actual Kaggle paths
- [ ] Ran Cell 1-6 successfully (audit shows class distribution)
- [ ] Class weights tensor looks reasonable (rare classes have higher weights)
- [ ] Internet is ON (for downloading DINOv2 backbone)

---

## 📞 NEED HELP?

If stuck:
1. Check Kaggle paths by running: `!ls -R /kaggle/input/`
2. Verify GPU: `!nvidia-smi`
3. Check dataset structure matches expected

---

**Ready? Start with Step 1 (creating ZIP) and Step 2 (upload to Kaggle).**
