# 👤 MEMBER 2 — WORK MAP & EXECUTION GUIDE
## Duality AI Hackathon | Augmentation + Class Weights + Report Lead

---

## 🎯 YOUR STRATEGY
> **Keep backbone FROZEN**. Focus on: Data diversity (augmentation) + Rare class handling (weights + copy-paste). This directly tackles the domain shift problem (train→test at different desert locations).

**Expected IoU Gain**: +20% to +30% over baseline (especially on rare classes)
**Target Val IoU**: 0.50–0.65

---

## 📊 WORK BREAKDOWN: NO DEPENDENCIES vs NEEDS OTHERS

### ✅ START IMMEDIATELY (No Dependencies)

| Priority | Task | Output | Time | Shares With |
|----------|------|--------|------|-------------|
| **P0** | Kaggle Setup | Working GPU env | 30 min | - |
| **P1** | `audit_dataset.py` | Class weights (**SHARE THIS FIRST!**) | 1-2h | Member 1 & 3 |
| **P2** | `augmentations.py` | 12-transform pipeline | 1h | Member 1 & 3 |
| **P3** | `dataset.py` | Unified dataset class | 1h | Member 1 & 3 |
| **P4** | `rare_class_tools.py` | Copy-paste augmentation | 1h | Member 1 |
| **P5** | Report Sections 1,3 | Title, Summary, Data Analysis | 2h | Team |

### ⏳ WAIT FOR YOUR OWN OUTPUTS (Depends on your work above)

| Priority | Task | Needs | Output | Time |
|----------|------|-------|--------|------|
| **P6** | Kaggle Training Notebook | Tasks P1-P4 complete | `model_augmented_best.pth` | 3h runtime |
| **P7** | Report Sections 2,5,6 | Training metrics | Methodology, Per-Class Analysis | 2h |

### 🤝 WAIT FOR TEAM (Needs others' results)

| Priority | Task | Needs | Output | Time |
|----------|------|-------|--------|------|
| **P8** | Report Section 4 | All 3 IoU scores | Ablation Study table | 30 min |
| **P9** | Final Report + Charts | Best model predictions | `REPORT.pdf` | 2h |

---

## 🗓️ DAY-BY-DAY SCHEDULE

### DAY 1 — Build Day (No training yet)

| Hour | Task | Deliverable |
|------|------|-------------|
| 0-0.5 | Kaggle setup + GPU check | Verified environment |
| 0.5-2.5 | **Run `audit_dataset.py`** | `class_distribution.png` + `class_weights.pt` |
| 2.5-3.5 | Write `augmentations.py` | 12-transform pipeline + visual check |
| 3.5-4.5 | Write `rare_class_tools.py` | Copy-paste pool builder |
| 4.5-5.5 | Write `dataset.py` | Unified dataset class |
| 5.5-6.5 | Prepare Kaggle Notebook 2 | Ready to train |
| 6.5-7.5 | Write Report Sections 1,3 | Title page + Data Analysis |

**Day 1 End Gate**: 
- [ ] `class_weights.pt` shared with team
- [ ] `augmentations.py` tested
- [ ] Kaggle Notebook ready for training

---

### DAY 2 — Training Day

| Hour | Task | Deliverable |
|------|------|-------------|
| 0-0.5 | Start Kaggle Notebook 2 | Training running |
| 0.5-3.5 | **Training runs** (50 epochs, ~3 hours) | Monitor via Kaggle logs |
| 3.5-5.5 | Fill idle time: Write Report Section 2 (Methodology) | Draft text |
| 5.5-6.5 | Download model, run quick val check | IoU score for ablation table |
| 6.5-7.5 | Share IoU with team, fill report templates | Updated report sections |

**Day 2 End Gate**:
- [ ] Training complete, model downloaded
- [ ] Val IoU reported to team
- [ ] Report 60% written

---

### DAY 3 — Integration + Final Report

| Hour | Task | Deliverable |
|------|------|-------------|
| 0-1 | Team sync: compare all 3 IoU scores | Ablation table filled |
| 1-3 | Finalize Report Sections 4,5,6,7,8 with real numbers | Complete draft |
| 3-4 | Generate final charts, confusion matrix | All visuals inserted |
| 4-5 | Proofread, export to PDF | `REPORT.pdf` (≤8 pages) |

**Day 3 End Gate**:
- [ ] Report PDF complete
- [ ] Ready for packaging

---

## 📁 YOUR FILE OWNERSHIP

```
Member 2 Files:
├── audit_dataset.py          ← RUN THIS FIRST (outputs class weights)
├── augmentations.py            ← 12-transform Albumentations pipeline
├── rare_class_tools.py         ← Copy-paste for Logs/Ground Clutter
├── dataset.py                  ← Unified dataset (uses augmentations)
├── member2_kaggle_notebook.py  ← Your complete training notebook
├── REPORT.pdf                  ← Final 8-page report (your main deliverable)
└── Outputs:
    ├── class_distribution.png
    ├── class_weights.pt        ← SHARE WITH TEAM
    ├── model_augmented_best.pth
    └── augmentation_curves.png
```

---

## ⚡ CRITICAL PATH FOR MEMBER 2

```
Hour 0:   Kaggle Setup ──────────────────────────────────────────┐
Hour 1:   audit_dataset.py → class_weights.pt ───────────────────┤
          (SHARE IMMEDIATELY WITH MEMBER 1 & 3)                   │
Hour 3:   augmentations.py + rare_class_tools.py ────────────────┤
Hour 5:   dataset.py + training notebook ──────────────────────┤
Hour 6:   Start training on Kaggle ──────────────────────────────┤
Day 2:    Training running (2-3 hours) ──────────────────────────┤
          → Fill time: Write report sections                        │
Day 2 PM: Training done → Report IoU ────────────────────────────┤
Day 3:    Finalize report with real numbers ─────────────────────┘
```

---

## 🚨 WHAT TO SHARE AND WHEN

| Time | What | Who Needs It | How |
|------|------|--------------|-----|
| **Day 1, Hour 2** | `class_weights.pt` + class distribution chart | Member 1 & 3 | Kaggle Dataset or team chat |
| **Day 1, Hour 5** | `augmentations.py` code | Member 1 (if they want to use it) | GitHub/team drive |
| **Day 2, Hour 6** | Your Val IoU score | All members | Team doc/chat |
| **Day 2, Hour 6** | `model_augmented_best.pth` | Member 3 (for ensemble testing) | Kaggle Dataset |

---

## ✅ SUCCESS CHECKLIST FOR MEMBER 2

### Before Training Starts (Day 1)
- [ ] Kaggle GPU working
- [ ] Dataset paths verified
- [ ] `audit_dataset.py` run → class weights computed
- [ ] Class weights shared with team
- [ ] Augmentations visually verified (mask alignment correct)
- [ ] Training notebook ready

### After Training (Day 2)
- [ ] `model_augmented_best.pth` downloaded
- [ ] Val IoU > 0.40 (if not, debug)
- [ ] Per-class IoU shows Logs > 0 (rare class detection working)
- [ ] Report sections 1,2,3,5,6 drafted

### Final (Day 3)
- [ ] All 3 member IoU scores collected
- [ ] Ablation table complete
- [ ] Report PDF ≤ 8 pages
- [ ] All charts inserted

---

## 💡 TIPS FOR EFFICIENCY

1. **Audit FIRST**: Run `audit_dataset.py` immediately — Member 1 needs those class weights to start their work

2. **Visual checks**: Always save 5-10 augmented images with masks overlay to verify alignment before training

3. **Fill idle time**: While training runs (3 hours), write report sections that don't need real numbers

4. **Backup everything**: Download checkpoints from `/kaggle/working/` every 10 epochs

5. **Communicate**: Post your IoU as soon as training ends — the team needs it to decide the final model

---

## 🔧 DETAILED TASK INSTRUCTIONS

### TASK 2A: Dataset Audit (`audit_dataset.py`)

**Purpose**: Analyze class distribution, compute weights for loss function.

**Steps**:
1. Scan all 2,857 training masks
2. Count pixels per class (raw IDs: 100, 200, 300...)
3. Remap to 0-9 indices
4. Compute weights: `1 / sqrt(frequency)`
5. Save `class_weights.pt` and `class_distribution.png`

**Output to share**:
```
class_weights.pt → Member 1 (for training)
class_weights.pt → Member 3 (for their experiments)
class_distribution.png → Team (for report)
```

---

### TASK 2B: Augmentation Pipeline (`augmentations.py`)

**Purpose**: Create 12-transform Albumentations pipeline for training diversity.

**Required transforms**:
1. `HorizontalFlip(p=0.5)`
2. `ShiftScaleRotate(shift=0.1, scale=0.3, rotate=15, p=0.7)`
3. `RandomResizedCrop(h=252, w=462, scale=(0.5,1.5), p=1.0)`
4. `Perspective(scale=(0.02, 0.05), p=0.2)`
5. `OneOf([ColorJitter, RandomGamma, CLAHE], p=0.5)`
6. `OneOf([GaussianBlur, MotionBlur], p=0.2)`
7. `GaussNoise(var_limit=(10,50), p=0.15)`
8. `RandomShadow(p=0.2)`
9. `RandomFog(fog_coef=(0.1,0.25), p=0.1)`
10. `RandomBrightnessContrast(p=0.3)`
11. `CoarseDropout(max_holes=6, max_h=40, max_w=40, p=0.15)`
12. `ToGray(p=0.05)`
13. `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`
14. `ToTensorV2()`

**Critical**: Use `additional_targets={'mask': 'mask'}` so masks get same spatial transforms!

---

### TASK 2D: Rare Class Tools (`rare_class_tools.py`)

**Purpose**: Copy-paste augmentation for Logs (0.07% of pixels) and Ground Clutter.

**Functions needed**:
- `build_rare_class_pool(mask_dir, image_dir, class_ids, min_area=100)` → extracts crop pool
- `CopyPasteAugmentor` class → applies copy-paste during training
- `WeightedRandomSampler` setup → 3x weight for images with rare classes

---

### TASK 2E: Ablation Baseline Training

**Your training approach** (from BUILD_CHECKLIST_UPDATED.md):
- SGD → AdamW (`lr=1e-4, weight_decay=0.01`)
- Add computed class weights to `CrossEntropyLoss(weight=...)`
- Epochs: 10 → 50
- Add `CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)`
- Add full augmentation pipeline from `augmentations.py`
- **NO backbone unfreezing, NO Lovász, NO decoder changes**
- Save: `model_augmented_best.pth`

---

## 📞 WHEN TO ASK FOR HELP

| Problem | Who to Ask | When |
|---------|------------|------|
| Mask values wrong (not 0-9) | Self-check: re-run audit | Day 1 |
| Augmentation misaligns masks | Self-check: add `additional_targets` | Day 1 |
| Class weights seem wrong | Member 1 for second opinion | Day 1 |
| Val IoU < 0.30 after 20 epochs | Team debug call | Day 2 |
| Training crashes / OOM | Member 1 (they're lead engineer) | Day 2 |
| Need IoU from others for report | Ping in team chat | Day 3 |

---

> **Member 2 Mission**: Make the model see MORE variety through augmentation and pay MORE attention to rare classes through weighting. This is how we beat the domain shift challenge.

---

## 🏃 LET'S START BUILDING

Next: I'll create `audit_dataset.py` — your first task that has no dependencies and provides critical output for the team.

