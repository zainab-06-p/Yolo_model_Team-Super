# DUALITY AI HACKATHON
## Offroad Semantic Scene Segmentation
### Full Team Plan — 3 Members, 3 Kaggle Notebooks

---

# PART 1: Simple Overview (Read This First)

---

## What Are We Building?

We are training an AI model that can look at a desert photo and label every single pixel — "that's a rock", "that's a tree", "that's the sky" etc. This is called **Semantic Segmentation**.

The model needs to work not just on photos it has seen, but also on **brand new desert locations** it has never seen before. That is the main challenge.

---

## What Is the Dataset?

Duality AI gives us two sets of images generated from their Falcon simulation platform:

- **Train + Val folder** — desert images WITH correct labels (we use these to train)
- **testImages folder** — desert images WITHOUT labels (judges evaluate our predictions on these)

---

## Our 3-Member Strategy

Each member trains the same base model but with a **different improvement technique**. We compare results and combine the best approach into one final model.

| Member | Technique | Goal |
|--------|-----------|------|
| Member 1 | Pretrained Model Fine-Tuning | Use DINOv2 backbone + unfreeze for better features |
| Member 2 | Data Augmentation + Class Weights | Help model generalize to new desert locations |
| Member 3 | Hyperparameter Tuning | Find best learning rate, scheduler, epochs |

---

## Why 3 Separate Approaches?

This is called an **Ablation Study** — it is what professional ML researchers do. We test each method individually so we can **scientifically prove** which one helped the most. This makes our report extremely strong.

| Approach | Expected IoU | What it proves |
|----------|-------------|----------------|
| Baseline (no changes) | 0.30 - 0.40 | Starting point to compare against |
| Fine-tuning only | 0.50 - 0.60 | How much pretrained backbone helps |
| Augmentation only | 0.50 - 0.65 | How much diversity helps generalization |
| Hyperparameter tuning | 0.40 - 0.55 | How much settings matter |
| **All combined (final)** | **0.70 - 0.80** | **Best possible model** |

---

## Simple Step-by-Step Flow

1. All 3 members download the dataset from Falcon to their Kaggle notebooks
2. Member 1 runs their modified training script (fine-tuning version)
3. Member 2 runs their modified training script (augmentation version)
4. Member 3 runs their modified training script (hyperparameter version)
5. All 3 run the test script and note down their IoU scores
6. Team compares IoU scores — pick the best approach
7. Combine best elements into one final training script
8. Train final model, run test script, save predictions
9. Write report using graphs and comparison images
10. Submit final model + report

---

# PART 2: Technical Details

---

## Understanding the Provided Scripts

Duality AI gave us 3 scripts. Here is what each one does:

| Script | What it does | When to run |
|--------|-------------|-------------|
| `train_segmentation.py` | Trains the DINOv2 + segmentation head model on desert images. Saves model weights, loss graphs, IoU scores. | First — before anything else |
| `test_segmentation.py` | Loads saved model, runs predictions on val/test images, saves colorized masks and per-class IoU scores. | After training is complete |
| `visualize.py` | Converts raw segmentation mask files into colorful easy-to-read images for the report. | Last — for report visuals |

---

## The Model Architecture

The scripts already use **DINOv2** — Facebook's powerful pretrained Vision Transformer. This is great news because we get transfer learning for free.

```
DINOv2 (pretrained backbone) — frozen by default, extracts features
         ↓
SegmentationHeadConvNeXt — small custom head, this is what gets trained
         ↓
Output: 10-class pixel prediction map
```

---

## The 10 Classes We Segment

| ID | Class Name | Difficulty |
|----|-----------|------------|
| 0 | Background | Easy |
| 100 → 1 | Trees | Medium |
| 200 → 2 | Lush Bushes | Medium |
| 300 → 3 | Dry Grass | Hard — looks like Landscape |
| 500 → 4 | Dry Bushes | Hard — looks like Dry Grass |
| 550 → 5 | Ground Clutter | Hard — few pixels |
| 700 → 6 | Logs | **Very Hard — rare in images** |
| 800 → 7 | Rocks | Medium |
| 7100 → 8 | Landscape | Easy — large area |
| 10000 → 9 | Sky | Easy — distinct blue |

---

## Key Hyperparameters in the Script

| Parameter | Default Value | Recommended Change |
|-----------|-------------|-------------------|
| `batch_size` | 2 | Keep at 2 for Kaggle free tier |
| `lr` (learning rate) | 1e-4 (0.0001) | Good for fine-tuning, keep same |
| `n_epochs` | 10 | Increase to 50 for better training |
| `optimizer` | SGD | Change to AdamW for better results |
| `loss function` | CrossEntropyLoss | Add class weights to fix imbalance |

---

# PART 3: Individual Member Plans

---

## Member 1 — Fine-Tuning (Kaggle Notebook 1)

### What to change:
- Unfreeze DINOv2 backbone so it also learns from our desert data
- Switch optimizer from SGD to AdamW
- Increase epochs from 10 to 50
- Add learning rate scheduler

### Key code changes:

```python
# Change 1: Unfreeze backbone
backbone_model.train()  # remove .eval()
for param in backbone_model.parameters():
    param.requires_grad = True

# Change 2: Better optimizer with separate LRs
optimizer = optim.AdamW([
    {'params': classifier.parameters(),   'lr': 1e-4},
    {'params': backbone_model.parameters(), 'lr': 1e-5},  # lower for pretrained
], weight_decay=0.01)

# Change 3: More epochs
n_epochs = 50

# Change 4: Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=n_epochs, eta_min=1e-6
)
```

### Expected output:
- Model saved as: `model_finetuned_best.pth`
- Expected IoU improvement: **+15% to +25%** over baseline

---

## Member 2 — Data Augmentation (Kaggle Notebook 2)

### What to change:
- Add Albumentations augmentation pipeline to training data
- Add class weights to loss function for rare classes
- Keep backbone frozen (only train the head)

### Key code changes:

```python
# Change 1: Install albumentations
!pip install albumentations

# Change 2: Augmentation pipeline
import albumentations as A

aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.4),
    A.GaussNoise(p=0.2),
    A.RandomFog(p=0.1),
], additional_targets={'mask': 'mask'})

# Change 3: Weighted loss for rare classes
class_weights = torch.tensor([
    1.0,  # Background
    2.0,  # Trees
    2.0,  # Lush Bushes
    1.5,  # Dry Grass
    2.0,  # Dry Bushes
    3.0,  # Ground Clutter
    5.0,  # Logs           ← rarest, highest weight
    3.0,  # Rocks
    1.0,  # Landscape
    1.0,  # Sky
]).to(device)

loss_fct = nn.CrossEntropyLoss(weight=class_weights)
```

### Expected output:
- Model saved as: `model_augmented_best.pth`
- Expected IoU improvement: **+20% to +30%** especially on rare classes

---

## Member 3 — Hyperparameter Tuning (Kaggle Notebook 3)

### What to change:
- Add CosineAnnealingLR learning rate scheduler
- Try different learning rates systematically
- Add early stopping to prevent overfitting
- Run multiple experiments and log which is best

### Key code changes:

```python
# Change 1: Define experiments to compare
EXPERIMENTS = [
    {"name": "exp1_baseline_adam",  "lr": 1e-4, "scheduler": "cosine"},
    {"name": "exp2_lower_lr",       "lr": 5e-5, "scheduler": "cosine"},
    {"name": "exp3_cyclic_lr",      "lr": 1e-4, "scheduler": "cyclic"},
]

# Change 2: Cosine scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=n_epochs, eta_min=1e-6
)

# Change 3: Early stopping
patience_counter = 0
early_stop_patience = 10

if val_iou > best_val_iou:
    best_val_iou = val_iou
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= early_stop_patience:
        print("Early stopping!")
        break
```

### Expected output:
- Best model saved as: `model_best.pth` inside best experiment folder
- Expected IoU improvement: **+5% to +15%** over baseline

---

# PART 4: Kaggle Setup Instructions

---

## Why Kaggle Over Google Colab?

| Feature | Kaggle | Google Colab (Free) |
|---------|--------|---------------------|
| Free GPU hours | 30 hrs/week per account | ~12 hrs/day but disconnects |
| Session stability | More stable, less disconnects | Disconnects after 90 min idle |
| Dataset upload | Dataset feature — no re-upload | Must re-upload every session |
| GPU type | Tesla P100 or T4 | T4 |
| Persistence | Notebook saves automatically | Must save to Drive manually |

---

## Step by Step Kaggle Setup (Do This Once)

1. Go to [kaggle.com](https://kaggle.com) and create a free account
2. Click **New Notebook**
3. On the right panel click **Add Data → Upload dataset** → upload the Duality dataset ZIP
4. Enable GPU: **Settings → Accelerator → GPU T4 x2**
5. Enable internet: **Settings → Internet → On** (needed to download DINOv2)
6. Paste the appropriate script for your member role
7. Click **Run All** and let it train

---

## Important Kaggle Paths

When you upload dataset to Kaggle, it goes to:
```
/kaggle/input/YOUR-DATASET-NAME/
```

Update dataset paths in the script:
```python
data_dir = '/kaggle/input/duality-desert/Offroad_Segmentation_Training_Dataset/train'
val_dir  = '/kaggle/input/duality-desert/Offroad_Segmentation_Training_Dataset/val'
```

---

## Saving Models on Kaggle

Always save to `/kaggle/working/` — this persists after the session:

```python
model_path = '/kaggle/working/segmentation_head.pth'
torch.save(classifier.state_dict(), model_path)
```

You can then download from the **Output tab** on the right panel.

---

# PART 5: Exact Running Order on Kaggle

---

## All 3 Members — Run Simultaneously

Each member opens their own Kaggle notebook and runs **independently at the same time**. This saves a lot of time.

| Step | What to Run | Expected Time |
|------|------------|---------------|
| 1 | Install dependencies cell | 2 minutes |
| 2 | Verify dataset paths cell | 1 minute |
| 3 | Run training script (`train_segmentation.py`) | 1–2 hours on GPU |
| 4 | Check `train_stats/` folder for loss graphs | 5 minutes |
| 5 | Run test script on val folder (`test_segmentation.py`) | 15–20 minutes |
| 6 | Note down IoU score from `evaluation_metrics.txt` | 1 minute |
| 7 | Run `visualize.py` for report images | 5 minutes |
| 8 | Share IoU score with team | Immediate |

---

## After All 3 Finish — Team Combines Results

1. Compare all 3 IoU scores in a table
2. Identify which method gave the biggest improvement
3. Combine best elements into one final script
4. One member trains the final combined model
5. Run `test_segmentation.py` one last time for final IoU
6. Download `predictions/` folder for submission

---

## Output Folder Structure

After running all scripts, your output folder looks like:

```
/kaggle/working/
├── segmentation_head.pth        ← trained model weights
├── train_stats/
│   ├── training_curves.png      ← loss + accuracy graphs
│   ├── iou_curves.png           ← IoU over epochs
│   ├── all_metrics_curves.png   ← combined metrics
│   └── evaluation_metrics.txt  ← final numbers
└── predictions/
    ├── masks/                   ← raw prediction files (class IDs 0–9)
    ├── masks_color/             ← colorized predictions (RGB)
    ├── comparisons/             ← side-by-side GT vs prediction images
    ├── evaluation_metrics.txt   ← per-class IoU scores
    └── per_class_metrics.png    ← bar chart of per-class IoU
```

---

# PART 6: Understanding Your Results

---

## IoU Score Targets

| IoU Score | Meaning | Action |
|-----------|---------|--------|
| 0.0 – 0.3 | Very Poor | Something is wrong — check data paths |
| 0.3 – 0.5 | Poor / Baseline | Normal starting point, improve further |
| 0.5 – 0.65 | Decent | Model is learning well |
| 0.65 – 0.75 | Good | Competitive submission |
| 0.75 – 0.85 | Very Good | Strong submission, likely top ranks |
| 0.85+ | Excellent | Top tier submission |

> **Target:** Get Val IoU above **0.70** before submitting.

---

## What to Put in Your Report

| Report Page | Section | What to include |
|-------------|---------|-----------------|
| 1 | Title | Team name, tagline, brief summary |
| 2 | Methodology | All 3 approaches, model architecture diagram |
| 3–4 | Results & Metrics | IoU comparison table, loss graphs, per-class IoU bar chart |
| 5–6 | Challenges & Solutions | Failure case images, what went wrong and how fixed |
| 7 | Conclusion | Best approach, final IoU, future improvements |

---

## Ablation Study Results Table (Fill This In)

Use this table in your report to show the comparison:

| Approach | Val IoU | Improvement vs Baseline | Key Insight |
|----------|---------|------------------------|-------------|
| Baseline | — | — | Starting point |
| Member 1: Fine-Tuning | — | — | |
| Member 2: Augmentation | — | — | |
| Member 3: Hyperparams | — | — | |
| **Final Combined** | — | — | **Best model** |

---

## Quick Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| IoU stuck near 0 | Wrong dataset paths | Check `/kaggle/input/` path carefully |
| CUDA out of memory | Batch size too large | Reduce `batch_size` to 1 |
| Loss not decreasing | Learning rate too high | Try `lr=1e-5` |
| Logs/Flowers IoU = 0 | Class imbalance | Add class weights to loss function |
| DINOv2 download fails | Internet not enabled | Kaggle Settings → Internet → On |
| Session disconnected | Idle timeout | Save checkpoint every 10 epochs to `/kaggle/working/` |

---

## Key Numbers to Remember

```
✅ IoU above 0.70    → Good submission
✅ Loss below 0.4    → Well trained model
✅ Inference < 50ms  → Real world ready
✅ LR = 0.0001       → For pretrained models
✅ Batch size = 2    → Safe for Kaggle free GPU
✅ 50 epochs         → Good training duration
```

---

> **Good luck to the team! Focus on IoU above 0.70 and a clean report.**
> 
> The ablation study approach will make your submission stand out. 🏆
