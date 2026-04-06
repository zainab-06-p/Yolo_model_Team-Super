# 📘 STEP-BY-STEP EXECUTION GUIDE
## Duality AI Hackathon | All 3 Members | Kaggle T4 GPU

---

## ⚡ BEFORE ANYTHING — ONE-TIME SETUP (All 3 members do this)

### Step 1: Create Kaggle Account & Enable GPU
1. Go to kaggle.com → Sign up / Log in
2. Click **"Create Notebook"** (top right)
3. On the right panel → **"Session options"** → Accelerator → select **GPU T4 x2**
4. Toggle **"Internet"** to ON (needed for DINOv2 download)
5. Click **Save**

### Step 2: Upload the Dataset
1. Go to kaggle.com → **Datasets** → **New Dataset**
2. Upload the Duality AI dataset ZIP file
3. Name it something like `duality-desert-seg`
4. Once uploaded, note the exact path — it will be:
   ```
   /kaggle/input/duality-desert-seg/
   ```
5. In your notebook → Add Data → search your dataset name → Add it

### Step 3: Update the Dataset Path in Your Notebook
In each notebook, find the `CONFIG` block at the top and update:
```python
'train_dir': '/kaggle/input/duality-desert-seg/Offroad_Segmentation_Training_Dataset/train',
'val_dir'  : '/kaggle/input/duality-desert-seg/Offroad_Segmentation_Training_Dataset/val',
'test_dir' : '/kaggle/input/duality-desert-seg/Offroad_Segmentation_testImages',
```
> ⚠️ The folder name after `/kaggle/input/` depends on what you named your dataset.
> Run `!ls /kaggle/input/` in a cell to see the exact name.

### Step 4: Verify Setup (run in any notebook)
```python
# Cell: Quick environment check
import torch, os, subprocess

print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU : {torch.cuda.get_device_name(0)}")
print(f"\nDataset contents:")
print(os.listdir('/kaggle/input/'))
```

---

## 👤 MEMBER 1 — Fine-Tuning Notebook (`member1_finetuning.py`)

### What this notebook does:
- Phase 1: Train only the segmentation head (backbone frozen) for 25 epochs
- Phase 2: Unfreeze last 4 DINOv2 blocks and fine-tune for 25 more epochs
- Uses composite phased loss (CE+Dice → CE+Dice → Focal+Dice)
- Saves: `model_finetuned_best.pth`

### Execution Steps:

**Step 1** — Paste the full `member1_finetuning.py` code into your Kaggle notebook (one big cell or multiple cells split at the `# CELL` markers)

**Step 2** — Update dataset paths in `CONFIG` block (see setup above)

**Step 3** — Run Cell 1 (GPU check) → verify you see T4 and CUDA=True

**Step 4** — Run Cell 4 (CONFIG) → verify output dir created

**Step 5** — Run Cell 6 (Dataset) → verify:
```
Train samples : 2857
Val samples   : 317
Batch shapes  : imgs=torch.Size([8, 3, 252, 462]), masks=torch.Size([8, 252, 462])
Mask unique   : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Dataset sanity check PASSED ✓
```
> If mask values are NOT 0–9, stop. Your VALUE_MAP or mask loading is wrong.

**Step 6** — Run Cell 10 (Load backbone) → wait ~2 min for DINOv2 to download

**Step 7** — Run Cell 13 (Phase 1 Training) → Expected time: **2–4 hours**
- Watch the first 3 epochs — loss should decrease
- Val mIoU should be > 0.10 after epoch 3
- If mIoU is stuck at 0.0: the mask values are wrong — re-check VALUE_MAP

**Step 8** — Check Phase 1 result:
```
✅ Phase 1 Complete! Best Val mIoU: 0.XXXX
Target was > 0.40 — PASSED ✓
```
> If < 0.40, message the team before starting Phase 2

**Step 9** — Run Cell 14 (Phase 2 Unfreeze) → Expected time: **2–3 hours**
- Batch size drops to 4 automatically (less VRAM needed)
- Watch for OOM error — if it happens, reduce batch size to 2

**Step 10** — After training completes:
```python
# Download your best model weights
from IPython.display import FileLink
FileLink('/kaggle/working/member1_outputs/model_finetuned_best.pth')
```
- Share the best Val mIoU number with the team

**Step 11** — Run Cell 15 (Report result) → post the mIoU to the team doc

### ⚠️ Common Issues for Member 1:
| Problem | Fix |
|---|---|
| CUDA OOM in Phase 2 | Reduce `phase2_batch_size` from 4 to 2 |
| DINOv2 download fails | Check Internet is ON in notebook settings |
| Loss is NaN from epoch 1 | Check mask values — run `masks.unique()` |
| Phase 1 mIoU stuck at 0.0 | Mask `*255` bug — verify masks are long tensors 0–9 |
| Kaggle session timeout | Checkpoints save every 5 epochs — use `--resume` logic |

---

## 👤 MEMBER 2 — Augmentation Notebook (`member2_augmentation.py`)

### What this notebook does:
- Runs dataset audit → computes class distribution and weights
- Trains with full Albumentations augmentation pipeline
- Uses WeightedRandomSampler to oversample rare class images
- Uses copy-paste augmentation for Logs and Ground Clutter
- Backbone stays frozen — all improvement comes from data
- Saves: `model_augmented_best.pth`

### Execution Steps:

**Step 1** — Paste `member2_augmentation.py` into your Kaggle notebook

**Step 2** — Update dataset paths in CONFIG

**Step 3** — Run Cell 1 (GPU check)

**Step 4** — Install Albumentations if needed:
```python
!pip install albumentations -q
```
Then run Cell 2 → verify:
```
Albumentations version: X.X.X ✓
```

**Step 5** — Run Cell 6 (Dataset Audit) → **run this FIRST before anything else**
Expected output:
```
📊 CLASS DISTRIBUTION:
Class                Pixels   Frequency   Weight
Background                0      0.000%    ...
Trees              12345678     15.234%    2.1x
...
Logs                  45678      0.07%    8.0x  ← rarest class
```
- Chart saved as `class_distribution.png`
- **Share the class_weights tensor with Member 1 via team chat**

**Step 6** — Run Cell 7 (Augmentation pipeline)

**Step 7** — Run Cell 8 (Dataset with Albumentations) → verify sanity check passes
- Also opens `augmentation_check.png` — visually confirm mask still aligns with image after augmentation

**Step 8** — Run Cell 9 (Copy-paste pool) → wait for pool to build:
```
Building copy-paste pool (target: 200 crops)...
Pool built: 187 crops ✓
```
> If pool has < 50 crops: rare classes might be absent in first N images — increase scan range

**Step 9** — Run Cell 10–11 (Model + Backbone)

**Step 10** — Run Cell 12 (Training) → Expected time: **2–3 hours**
- Watch per-class IoU at each improvement — are Logs IoU > 0 after 10 epochs?
- If Logs IoU stays 0.0: copy-paste isn't working — verify `RARE_CLASSES = [5, 6]`

**Step 11** — After training:
```python
from IPython.display import FileLink
FileLink('/kaggle/working/member2_outputs/model_augmented_best.pth')
FileLink('/kaggle/working/member2_outputs/class_distribution.png')
```

**Step 12** — Run Cell 13 → post mIoU to team doc

### While Training Runs — Write These Report Sections:
- **Section 1**: Title, team name, one-paragraph summary
- **Section 3**: Data Analysis — insert `class_distribution.png`, explain class imbalance
- **Section 6**: Challenges & Solutions — class imbalance → copy-paste + weighting

### ⚠️ Common Issues for Member 2:
| Problem | Fix |
|---|---|
| Albumentations mask misalignment | Add `additional_targets={'mask':'mask'}` to Compose |
| CoarseDropout crashes | Update albumentations: `!pip install -U albumentations` |
| WeightedRandomSampler slow | Reduce `sample_limit` in audit to 300 |
| Copy-paste pool is empty | Check `RARE_CLASSES` indices match VALUE_MAP remapped IDs |

---

## 👤 MEMBER 3 — Experiments + Inference Notebook (`member3_hyperparams_inference.py`)

### What this notebook does:
- Runs 3 hyperparameter experiments (different LR + scheduler combos)
- Each experiment uses early stopping — bad configs die fast
- Builds full TTA inference pipeline (4 variants, averaged probs)
- Runs final inference on all 1,002 test images
- Saves: raw masks, colorized masks, comparison images, per-class IoU chart

### Execution Steps:

**Step 1** — Paste `member3_hyperparams_inference.py` into your Kaggle notebook

**Step 2** — Update dataset paths in CONFIG

**Step 3** — Run Cell 1 (GPU check) + Cell 2 (imports)

**Step 4** — Run Cell 4 (Dataset) → verify sanity check

**Step 5** — Run Cell 6 (Load backbone) → wait for DINOv2 download

**Step 6** — Run Cell 7 (Experiments) → Expected time: **3–5 hours total**
- 3 experiments run back to back
- Each has early stopping (patience=10) so bad ones die quickly
- Results table printed at end:
```
HYPERPARAMETER EXPERIMENT RESULTS
Experiment                       Best Val mIoU
exp2_cosine_lr5e5                       0.5234  ← WINNER
exp1_cosine_lr1e4                       0.5011
exp3_onecycle_lr1e4                     0.4876
```

**Step 7** — After experiments: run Cell 8 (TTA) — no training, just defines the class

**Step 8** — Run Cell 9 (Test inference):
- Set `BEST_MODEL_PATH` to the winner experiment's model OR to Member 1's model if their mIoU is higher
- Runs on all 1,002 test images with TTA
- Verify: `saved = 1002` at the end

**Step 9** — Run Cell 10 (Val evaluation):
```
Without TTA : 0.5234
With TTA    : 0.5489
TTA boost   : +0.0255
```
- Note both numbers for the report

**Step 10** — Run Cell 11 (Summary) → post experiment results to team doc

**Step 11** — Download outputs:
```python
# Download the predictions folder as a zip
import shutil
shutil.make_archive('/kaggle/working/predictions_backup', 'zip',
                    '/kaggle/working/member3_outputs/predictions')
from IPython.display import FileLink
FileLink('/kaggle/working/predictions_backup.zip')
```

### While Experiments Run — Write These Report Sections:
- **Section 6**: Challenges & Solutions (use known issues: class imbalance, domain shift, OOM)
- **Section 7**: Failure Cases template (leave image slots empty — fill after inference)
- **requirements.txt**: list all packages used

### ⚠️ Common Issues for Member 3:
| Problem | Fix |
|---|---|
| Exp 3 OneCycle crashes | OneCycle needs `steps_per_epoch` — verify train_loader length > 0 |
| TTA dimension mismatch | Check all 4 variants return same [B, C, H, W] shape |
| Inference count ≠ 1002 | Some test images may have different extension — check with `!ls test_dir` |
| Scale 1.25× OOM | Remove scale variants, use only original + h-flip (2 variants) |
| Model weights mismatch | Architecture in test script must EXACTLY match training architecture |

---

## 🤝 DAY 3 — COMBINING ALL 3 RESULTS

### Step 1: Compare on Team Call
Fill the ablation table with real numbers:
```
Baseline (from original script): ~0.31
Member 1 (fine-tuning):          ___
Member 2 (augmentation):         ___
Member 3 (best experiment):      ___
```

### Step 2: Train Final Combined Model
Whoever got the highest mIoU trains the final model incorporating all improvements:
- Member 1's fine-tuning strategy (unfreeze last 4 blocks)
- Member 2's augmentation pipeline + class weights
- Member 3's best LR and scheduler settings
- All combined in one notebook

### Step 3: Run Final Inference with TTA
Use Member 3's inference pipeline with the final combined model weights.

### Step 4: Package Submission
```
submission/
├── train.py (member1 notebook as .py)
├── test.py  (member3 inference cells as .py)
├── model_final_best.pth
├── predictions/
│   ├── masks/         ← 1,002 files
│   ├── masks_color/   ← 1,002 files
│   └── comparisons/   ← 10 files
├── train_stats/
│   ├── training_curves_phase1.png
│   ├── training_curves_phase2.png
│   └── per_class_iou.png
├── REPORT.pdf
└── README.md
```

---

## ✅ FINAL GATE CHECKLIST (before submitting)

Run these checks in Member 3's notebook:
```python
import os

pred_dir = '/kaggle/working/member3_outputs/predictions/masks'
files = os.listdir(pred_dir)

print(f"Prediction count  : {len(files)}")
print(f"Expected          : 1002")
assert len(files) == 1002, "❌ Wrong count!"

# Check a mask
import numpy as np
from PIL import Image
sample = np.array(Image.open(os.path.join(pred_dir, files[0])))
print(f"Mask shape        : {sample.shape}")
print(f"Mask unique vals  : {np.unique(sample).tolist()}")
assert sample.max() <= 9, "❌ Mask values out of range!"

print("\n✅ All checks passed — ready to submit!")
```

---

> 🏁 Good luck team. The model that wins is the one submitted on time with clean code.
