# 👥 WORK DIVISION — REDESIGNED
## Duality AI Hackathon | Parallel Pipeline with Zero Blocking

> **Design principle**: No one waits idle. If your primary task depends on someone else,
> you have a pre-assigned secondary task to fill that gap. Every hour of every day is
> accounted for across all 3 members.

---

## 🔍 DEPENDENCY MAP (before redesigning)

First, let's be honest about what blocks what:

```
audit_dataset.py (M2)
    └──→ class_weights output
              └──→ losses.py needs class_weights        ← M1 BLOCKED by M2
              └──→ train.py needs class_weights         ← M1 BLOCKED by M2

dataset.py (M2)
    └──→ train.py imports dataset.py                   ← M1 BLOCKED by M2
    └──→ augmentations.py is part of dataset.py        ← M1 BLOCKED by M2

losses.py (M1)
    └──→ train.py imports losses.py                    ← M1 self-dependency (fine)

train.py (M1)
    └──→ produces model weights (.pth)
              └──→ test.py needs weights               ← M3 BLOCKED by M1
              └──→ visualize.py needs weights          ← M3 BLOCKED by M1
              └──→ Report needs real numbers           ← M2 BLOCKED by M1+M3

config.yaml (M1)
    └──→ ALL scripts import config                     ← M2, M3 BLOCKED by M1
```

### Root Problem in Original Plan
- M1 (training) depends on M2's dataset.py and class_weights → M1 can't start training
- M3 (test.py, visualize) depends on trained weights → M3 has nothing to do on Day 1-2
- M2 (report) depends on real numbers from ALL training → M2 writes a blank report until Day 3
- Config.yaml is owned by M1 but everyone needs it → creates a single point of failure

---

## ✅ REDESIGNED OWNERSHIP TABLE

Reassigning ownership to eliminate blocking:

| File / Task | OLD Owner | NEW Owner | Reason for Change |
|---|---|---|---|
| `config.yaml` | M1 | **M1 — but shared on Day 1 morning** | Must be done FIRST, everyone unblocked |
| `dataset.py` | M2 | **M2** (no change) | Core data work stays with M2 |
| `augmentations.py` | M2 | **M2** (no change) | Paired with dataset |
| `audit_dataset.py` | M2 | **M2** (no change, but run FIRST) | Outputs class weights for M1 |
| `losses.py` | M1 | **M1** (no change) | Self-contained, no dependency |
| `train.py` | M1 | **M1** (no change) | Core training |
| `rare_class_tools.py` | M2 | **M3** | M3 has idle time on Day 1 |
| `test.py` | M3 | **M3** (no change) | Inference |
| `tta.py` | M3 | **M3** (no change) | Pairs with test.py |
| `visualize.py` | M3 | **M3** (no change) | Post-training |
| `README.md` | M3 | **M3** (no change) | Packaging |
| `REPORT.pdf` | M2 | **M2** (no change) | Writing |
| Report skeleton + charts | M2 | **M2 starts Day 1** | Write what you CAN before numbers arrive |
| Backbone experiment (DINOv2-Base) | M3 | **M3** | Idle time filler on Day 2 |

---

## 🗓️ HOUR-BY-HOUR PARALLEL PIPELINE

### ═══ DAY 1 — Build Day ═══

```
HOUR      MEMBER 1                    MEMBER 2                    MEMBER 3
────────  ──────────────────────────  ──────────────────────────  ──────────────────────────
0–1h      🔴 ENV SETUP (all 3 together on call — verify GPU, dataset paths, DINOv2 loads)

1–2h      Write config.yaml           Run audit_dataset.py        ENV SETUP + explore dataset
          (share immediately          (class distribution,         manually (open images,
          in team chat when done)     class weights output)        open masks, verify sizes)

2–3h      Write losses.py             Write dataset.py            Write test.py skeleton
          (Lovász, Dice, Focal,       (Dataset class, __getitem__, (arg parsing, load model,
          phased loss — completely    mask remapping, 16-bit PNG   forward pass, save masks —
          independent, no blocks)     loading)                     no weights needed yet)

3–4h      Write SegmentationHead      Write augmentations.py      Write tta.py
          decoder (ConvNeXt stem,     (all 12 transforms, val     (4 TTA variants, prob
          independent of dataset)     transform, visual verify)    averaging — standalone)

4–5h      ⏳ WAITING on M2's          Write rare_class_tools.py   Write visualize.py
          dataset.py + weights        (build_rare_class_pool,      (confusion matrix,
          → FILL TIME:                copy_paste_augment,          per-class IoU bar chart,
          Write train.py SHELL        WeightedRandomSampler)       failure case gallery)
          (loop structure, AMP,
          grad clip, checkpointing
          — use placeholder for
          dataset import)

5–6h      ✅ M2 finishes dataset.py   ✅ Finish rare_class_tools   Write README.md skeleton
          → M1 plugs in dataset       Share class_weights.pt       (commands, structure —
          import + class weights      to M1 via team chat          fill real numbers later)
          → Complete train.py

6–7h      🔥 INTEGRATION TEST         🔥 INTEGRATION TEST          🔥 INTEGRATION TEST
          (all 3 together — run 1 training epoch end-to-end, verify shapes, no crash,
           loss decreasing, GPU >80%, mask values 0-9 correct)
           ✅ GATE 1 passed → Day 1 complete
```

---

### ═══ DAY 2 — Training Day ═══

```
HOUR      MEMBER 1                    MEMBER 2                    MEMBER 3
────────  ──────────────────────────  ──────────────────────────  ──────────────────────────
0–2h      🚀 START TRAINING           🚀 START TRAINING           🚀 START TRAINING
          Kaggle Notebook 1:          Kaggle Notebook 2:          Kaggle Notebook 3:
          Phase 1 frozen backbone     Augmentation + weights       Exp 1: cosine lr=1e-4
          25 epochs (~2-4 hrs)        50 epochs (~2-3 hrs)        50 epochs (~2-3 hrs)

          ⏳ TRAINING RUNNING         ⏳ TRAINING RUNNING         ⏳ TRAINING RUNNING
          → FILL TIME:                → FILL TIME:                → FILL TIME:
          Write report Section 2      Write report Section 1      Write report Section 6
          (Methodology — arch         (Title + Summary —          (Challenges & Solutions
          diagram, loss design,       team intro, one-para        — known issues: class
          two-phase plan)             approach summary)           imbalance, domain shift)

          [~3hrs in]                  [~2hrs in]                  [~2hrs in]
          Check training logs —       Training DONE ✅             Exp 1 DONE ✅
          is loss decreasing?         Download model_aug.pth      Record Exp 1 IoU
          Phase 1 continues...        Run quick val IoU check     → START Exp 2: lr=5e-5

                                      → FILL TIME:                ⏳ Exp 2 running
                                      Write report Section 3      → FILL TIME:
                                      (Data Analysis — use        Write report Section 7
                                      class_distribution.png      (Failure Cases — write
                                      already generated)          template, fill images later)

          Phase 1 DONE ✅             [async] M2 sends M1        Exp 2 DONE ✅
          Download phase1 weights     val IoU number              Record Exp 2 IoU
          → START Phase 2:                                        → START Exp 3: OneCycle
          unfreeze 4 blocks,
          batch=4, 25 epochs

          ⏳ Phase 2 running          → FILL TIME:                ⏳ Exp 3 running
          → FILL TIME:               Write report Section 5      → FILL TIME:
          Review all code for        (Per-Class Analysis —       Write packaging checklist
          Gate 3 items               template, fill confusion    (requirements.txt,
          (class mapping check,      matrix + images later)      folder structure)
          config matches trained)

          Phase 2 DONE ✅            Exp 3 DONE ✅
          Download best_model.pth    Record all 3 IoU scores
          Share with team            → Pick best experiment
```

---

### ═══ DAY 3 — Integration + Inference Day ═══

```
HOUR      MEMBER 1                    MEMBER 2                    MEMBER 3
────────  ──────────────────────────  ──────────────────────────  ──────────────────────────
0–1h      🤝 TEAM SYNC — compare all 3 IoU scores, fill ablation table, decide final model

1–3h      Train FINAL COMBINED        Write report Section 4      Run test.py on VAL SET
          model (best settings        (Ablation Study — now you   with best available weights
          from all 3 experiments)     have real numbers to fill   Record: mIoU with/without
          All improvements merged:    into the table)             TTA
          - Unfrozen backbone (M1)
          - Augmentation (M2)
          - Best LR/scheduler (M3)

          ⏳ Final training running   Run visualize.py on M3's    Run test.py on TEST SET
          → FILL TIME:               best val predictions:        (1,002 images) → save
          Code review — verify        - confusion matrix          predictions/ folder
          class mapping identical     - per-class IoU chart       Verify: exactly 1,002 files
          in ALL files                - failure case gallery      Visual spot-check 10 images

          Final model DONE ✅         Insert real charts into     Run TTA on test set ✅
          Share final weights         report sections 4, 5, 7     Save colorized masks

3–4h      Final val IoU check         Write report Section 8      Build submission folder
          Final test predictions      (Conclusion & Future Work)  structure
          Visual verification         Proofread all 8 pages       Verify dry-run works

4–5h      🔥 FINAL INTEGRATION CHECK (all 3 together)
          ✅ Gate 3: all 1,002 predictions exist
          ✅ Class mapping identical across all files
          ✅ Report PDF ≤ 8 pages
          ✅ README commands work
          ✅ Dry-run: fresh notebook loads model + runs test.py → works
          ✅ No test images in training (verify by checking data paths)
          → SUBMIT ✅
```

---

## 📊 REVISED MEMBER PROFILES

### 👤 MEMBER 1 — Core ML Engineer
**Primary responsibility**: Model architecture + training pipeline + final combined model

**Why they're never blocked:**
- Day 1: `config.yaml` → `losses.py` → `SegmentationHead` are all 100% self-contained
- The only dependency is `dataset.py` from M2 — the `train.py` shell is written first,
  M2's dataset is plugged in when ready (~5h mark on Day 1)
- Day 2: Training runs autonomously on Kaggle — idle time used for report writing

**Files owned:**
```
config.yaml          ← done FIRST, shared immediately
losses.py            ← Lovász, Dice, Focal, phased switching
train.py             ← full training loop, both phases
segmentation_head.py ← ConvNeXt decoder
model_finetuned_best.pth  ← output
```

**Free-time tasks (while training runs):**
- Report Section 2 (Methodology)
- Code review for Gate 3 items
- Class mapping audit across all files

---

### 👤 MEMBER 2 — Data Engineer + Report Lead
**Primary responsibility**: Data pipeline + augmentation + report

**Why they're never blocked:**
- Day 1: `audit_dataset.py` runs first (produces class weights for M1, unblocking them)
- Then `dataset.py` → `augmentations.py` → own training — entirely self-contained
- Report writing starts Day 1 with what's already known (architecture, data analysis)
  and fills in real numbers as they arrive from M1 and M3

**Files owned:**
```
audit_dataset.py     ← run FIRST on Day 1, outputs class_weights.pt
dataset.py           ← Dataset class, mask loading, remapping
augmentations.py     ← full 12-transform pipeline
model_augmented_best.pth  ← output
REPORT.pdf           ← 8-page report
```

**Report writing schedule (fills idle time):**
```
Day 1 idle:    Section 1 (Title/Summary), Section 3 (Data Analysis)
Day 2 idle:    Section 2 (Methodology), Section 5 (Per-Class Analysis template)
Day 3:         Sections 4, 6, 7, 8 — fill with real numbers from M1+M3
               Proofread → export PDF
```

---

### 👤 MEMBER 3 — Inference Engineer + Packaging Lead
**Primary responsibility**: Hyperparameter search + test pipeline + final packaging

**Why they're never blocked:**
- Day 1: `test.py`, `tta.py`, `visualize.py`, `README.md` are ALL independent of model weights
  — they can be written against a dummy model and plugged in later
- Day 2: 3 experiments run sequentially on Kaggle — idle time used for report + packaging
- Day 3: inference runs using weights from M1+M2

**Files owned:**
```
test.py              ← inference, val metrics, test predictions
tta.py               ← 4-variant TTA wrapper
visualize.py         ← confusion matrix, failure cases, IoU charts
rare_class_tools.py  ← copy-paste pool, WeightedRandomSampler
README.md            ← step-by-step instructions
requirements.txt     ← all dependencies
```

**Free-time tasks (while experiments run):**
```
Day 1 idle:    Report Section 6 (Challenges & Solutions)
Day 2 idle:    Report Section 7 (Failure Cases template), requirements.txt
Day 3:         Final packaging, dry-run, submission folder
```

---

## ⚡ CRITICAL PATH (the sequence that determines how fast you finish)

```
Day 1, Hour 0:   ENV SETUP (all 3) ─────────────────────────────────┐
Day 1, Hour 1:   M2 runs audit_dataset.py → class_weights.pt ───────┤
Day 1, Hour 1:   M1 writes config.yaml → shared immediately ────────┤
Day 1, Hour 5:   M2 finishes dataset.py → M1 plugs in ─────────────┤
Day 1, Hour 6:   Integration test (1 epoch, all 3) ─────────────────┤
Day 2, Hour 0:   All 3 start training simultaneously ───────────────┤
Day 2, Hour 8:   All 3 training done → share IoU numbers ───────────┤
Day 3, Hour 0:   Sync → pick best → train final model ──────────────┤
Day 3, Hour 3:   M3 runs test predictions on test set ──────────────┤
Day 3, Hour 4:   Gate 3 check → SUBMIT ─────────────────────────────┘
```

Everything feeds into this chain. If any step slips, the whole team knows immediately.

---

## 🚨 BLOCKING SCENARIOS + RESOLUTIONS

| Scenario | Who's blocked | Resolution |
|---|---|---|
| M2's dataset.py takes longer than expected | M1 can't finish train.py | M1 writes train.py with a simple torchvision dummy dataset as placeholder. Swap in M2's when ready. |
| M1's training crashes in Phase 2 | Final model delayed | M2's augmentation model becomes primary submission. M1's Phase 1 weights used as fallback. |
| M3's experiments all give similar IoU | No clear best config | Combine all 3 configs into final model — every improvement stacks. |
| M2's report needs real numbers but M1/M3 still training | Report blocked | Write all text sections first. Leave number/chart placeholders. Fill in last 2 hours of Day 3. |
| Kaggle session dies mid-training | Training lost | Always `--resume` flag. Checkpoint every 5 epochs to `/kaggle/working/`. Download .pth after each checkpoint. |
| M3's test.py has shape mismatch with M1's model | Inference broken | M1 exports a `model_info.txt` with exact input/output shapes after training. M3 writes test.py to match. |

---

## 📋 COMMUNICATION PROTOCOL

To keep this parallel work from diverging:

```
Shared team doc (Google Doc or Notion):
  ├── config.yaml (live copy — M1 updates, everyone reads)
  ├── class_weights.pt path (M2 posts when done)
  ├── IoU scoreboard (everyone fills in as results come in)
  └── Ablation table (fill live during Day 2-3)

Sync calls:
  ├── Day 1, Hour 0: ENV SETUP (30 min, all 3)
  ├── Day 1, Hour 6: Integration test (30 min, all 3)
  ├── Day 2, Hour 0: Training kickoff (15 min, all 3)
  ├── Day 3, Hour 0: Results sync + final model decision (30 min, all 3)
  └── Day 3, Hour 4: Gate 3 + submit (30 min, all 3)

File sharing:
  → Use Kaggle Datasets to share .pth weights between members
  → Use GitHub or Google Drive for code files
  → Name convention: member1_phase1_epoch25.pth, member2_aug_best.pth, etc.
```

---

## 📊 WORK BALANCE CHECK

| Member | Day 1 Coding | Day 2 Active | Day 3 Active | Report | Total Load |
|---|---|---|---|---|---|
| M1 | Heavy (config, losses, train.py, head) | Medium (training + monitor) | Medium (final model) | Light (Section 2) | ⚖️ Balanced |
| M2 | Heavy (audit, dataset, augmentations) | Medium (training + monitor) | Light (visualize, proofread) | Heavy (all 8 pages) | ⚖️ Balanced |
| M3 | Medium (test, tta, visualize, rare_class) | Medium (3 experiments) | Heavy (inference + packaging) | Light (Sections 6,7) | ⚖️ Balanced |

---

> **Golden rule**: If you finish your task early, check the team doc for what's next.
> If you're blocked, switch to your free-time task immediately — never sit idle.
