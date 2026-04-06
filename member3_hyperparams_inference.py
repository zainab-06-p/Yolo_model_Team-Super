# ╔══════════════════════════════════════════════════════════════════════╗
# ║  MEMBER 3 — Hyperparameter Experiments + Inference + TTA            ║
# ║  Strategy: Find best LR/scheduler + build test pipeline             ║
# ║  Duality AI Hackathon | Kaggle T4 GPU                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ============================================================
# CELL 1 — GPU CHECK
# ============================================================
import subprocess
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
import torch
print(f"CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# CELL 2 — IMPORTS
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os, cv2, random
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
# CELL 3 — CONFIGURATION
# ============================================================
CONFIG = {
    'train_dir' : '/kaggle/input/YOUR-DATASET/Offroad_Segmentation_Training_Dataset/train',
    'val_dir'   : '/kaggle/input/YOUR-DATASET/Offroad_Segmentation_Training_Dataset/val',
    'test_dir'  : '/kaggle/input/YOUR-DATASET/Offroad_Segmentation_testImages',
    'output_dir': '/kaggle/working/member3_outputs',

    'img_w': 462,
    'img_h': 252,
    'n_classes'  : 10,
    'batch_size' : 8,
    'weight_decay': 0.01,
    'gradient_clip': 1.0,
    'early_stop_patience': 10,
    'n_epochs'   : 50,
    'num_workers': 2,
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

VALUE_MAP = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}
CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

# ── Color palette for visualizations ────────────────────────
COLOR_PALETTE = np.array([
    [0,   0,   0  ],  # Background - black
    [34,  139, 34 ],  # Trees - forest green
    [0,   255, 0  ],  # Lush Bushes - lime
    [210, 180, 140],  # Dry Grass - tan
    [139, 90,  43 ],  # Dry Bushes - brown
    [128, 128, 0  ],  # Ground Clutter - olive
    [139, 69,  19 ],  # Logs - saddle brown
    [128, 128, 128],  # Rocks - gray
    [160, 82,  45 ],  # Landscape - sienna
    [135, 206, 235],  # Sky - sky blue
], dtype=np.uint8)

print("Configuration ready ✓")

# ============================================================
# CELL 4 — DATASET (minimal — no albumentations needed here)
# ============================================================
h, w = CONFIG['img_h'], CONFIG['img_w']

def convert_mask_np(mask_pil):
    arr = np.array(mask_pil)
    out = np.zeros_like(arr, dtype=np.uint8)
    for raw_val, class_id in VALUE_MAP.items():
        out[arr == raw_val] = class_id
    return out

img_transform = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
mask_transform = transforms.Compose([
    transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])


class DesertDataset(Dataset):
    def __init__(self, data_dir, img_t=None, mask_t=None, has_masks=True):
        self.image_dir  = os.path.join(data_dir, 'Color_Images')
        self.mask_dir   = os.path.join(data_dir, 'Segmentation') if has_masks else None
        self.img_t      = img_t
        self.mask_t     = mask_t
        self.has_masks  = has_masks
        self.filenames  = sorted(os.listdir(self.image_dir))

    def __len__(self): return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img   = Image.open(os.path.join(self.image_dir, fname)).convert('RGB')
        if self.img_t: img = self.img_t(img)

        if self.has_masks:
            mask = Image.open(os.path.join(self.mask_dir, fname))
            mask = Image.fromarray(convert_mask_np(mask))
            if self.mask_t:
                mask = (self.mask_t(mask) * 255).long().squeeze(0)
            return img, mask, fname
        return img, fname


train_dataset = DesertDataset(CONFIG['train_dir'], img_transform, mask_transform)
val_dataset   = DesertDataset(CONFIG['val_dir'],   img_transform, mask_transform)
test_dataset  = DesertDataset(CONFIG['test_dir'],  img_transform, has_masks=False)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True,
                          num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_dataset,   batch_size=CONFIG['batch_size'], shuffle=False,
                          num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True)
test_loader  = DataLoader(test_dataset,  batch_size=4, shuffle=False,
                          num_workers=CONFIG['num_workers'], pin_memory=True)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# ── Sanity check ─────────────────────────────────────────────
imgs, masks, _ = next(iter(train_loader))
print(f"Batch: imgs={imgs.shape}, masks={masks.shape}")
assert masks.max() <= 9, "❌ Mask values out of range!"
print("Dataset sanity check PASSED ✓")

# ============================================================
# CELL 5 — MODEL (identical to Member 1 — must match!)
# ============================================================
class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH, dropout=0.1):
        super().__init__()
        self.tokenH, self.tokenW = tokenH, tokenW
        self.stem   = nn.Sequential(nn.Conv2d(in_channels, 256, 7, padding=3), nn.GELU())
        self.block1 = nn.Sequential(
            nn.Conv2d(256, 256, 7, padding=3, groups=256), nn.GELU(),
            nn.Conv2d(256, 256, 1), nn.GELU(),
        )
        self.block2    = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.GELU())
        self.dropout   = nn.Dropout2d(p=dropout)
        self.classifier= nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.tokenH, self.tokenW, C).permute(0, 3, 1, 2)
        x = self.stem(x); x = self.block1(x); x = self.block2(x)
        return self.classifier(self.dropout(x))


def compute_iou(logits, targets, n_classes=10):
    preds   = torch.argmax(logits, dim=1).view(-1)
    targets = targets.view(-1)
    ious = []
    for c in range(n_classes):
        pc = preds == c; tc = targets == c
        inter = (pc & tc).sum().float(); union = (pc | tc).sum().float()
        ious.append(float('nan') if union == 0 else (inter/union).item())
    return np.nanmean(ious), ious


print("Model defined ✓")

# ============================================================
# CELL 6 — LOAD BACKBONE
# ============================================================
print("Loading DINOv2 backbone...")
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
backbone = backbone.to(device).eval()
for p in backbone.parameters(): p.requires_grad = False

with torch.no_grad():
    sample = torch.zeros(1,3,h,w).to(device)
    feats  = backbone.forward_features(sample)['x_norm_patchtokens']
    n_emb  = feats.shape[2]
    tokenH, tokenW = h//14, w//14

print(f"Backbone loaded | Embedding: {n_emb} | Tokens: {tokenH}×{tokenW} ✓")

# ============================================================
# CELL 7 — HYPERPARAMETER EXPERIMENTS
# ============================================================
EXPERIMENTS = [
    {
        'name'     : 'exp1_cosine_lr1e4',
        'lr'       : 1e-4,
        'scheduler': 'cosine',
        'n_epochs' : CONFIG['n_epochs'],
        'patience' : CONFIG['early_stop_patience'],
    },
    {
        'name'     : 'exp2_cosine_lr5e5',
        'lr'       : 5e-5,
        'scheduler': 'cosine',
        'n_epochs' : CONFIG['n_epochs'],
        'patience' : CONFIG['early_stop_patience'],
    },
    {
        'name'     : 'exp3_onecycle_lr1e4',
        'lr'       : 1e-4,
        'scheduler': 'onecycle',
        'n_epochs' : CONFIG['n_epochs'],
        'patience' : CONFIG['early_stop_patience'],
    },
]

exp_results = {}

for exp in EXPERIMENTS:
    name = exp['name']
    exp_dir = os.path.join(CONFIG['output_dir'], name)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"LR: {exp['lr']} | Scheduler: {exp['scheduler']}")
    print(f"{'='*60}")

    # Fresh model for each experiment
    classifier = SegmentationHeadConvNeXt(n_emb, CONFIG['n_classes'], tokenW, tokenH).to(device)
    optimizer  = optim.AdamW(classifier.parameters(), lr=exp['lr'],
                              weight_decay=CONFIG['weight_decay'])

    if exp['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=exp['n_epochs'], eta_min=1e-6)
    elif exp['scheduler'] == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=exp['lr'],
            steps_per_epoch=len(train_loader),
            epochs=exp['n_epochs'])

    loss_fct = nn.CrossEntropyLoss()
    scaler   = GradScaler()

    history     = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'lr': []}
    best_iou    = 0.0
    patience_ctr= 0

    for epoch in range(exp['n_epochs']):
        # Train
        classifier.train()
        t_losses = []
        pbar = tqdm(train_loader, desc=f"[{name}] Ep {epoch+1}/{exp['n_epochs']}", leave=False)
        for imgs, masks, _ in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.no_grad():
                feats = backbone.forward_features(imgs)['x_norm_patchtokens']
            optimizer.zero_grad()
            with autocast():
                logits  = classifier(feats)
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode='bilinear', align_corners=False)
                loss    = loss_fct(outputs, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), CONFIG['gradient_clip'])
            scaler.step(optimizer); scaler.update()
            if exp['scheduler'] == 'onecycle': scheduler.step()
            t_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if exp['scheduler'] == 'cosine': scheduler.step()

        # Validate
        classifier.eval()
        v_losses, v_ious = [], []
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                feats   = backbone.forward_features(imgs)['x_norm_patchtokens']
                logits  = classifier(feats)
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode='bilinear', align_corners=False)
                v_losses.append(loss_fct(outputs, masks).item())
                iou, _  = compute_iou(outputs, masks)
                v_ious.append(iou)

        val_iou = np.nanmean(v_ious)
        history['train_loss'].append(np.mean(t_losses))
        history['val_loss'].append(np.mean(v_losses))
        history['val_iou'].append(float(val_iou))
        history['lr'].append(optimizer.param_groups[0]['lr'])

        print(f"  Ep {epoch+1:02d} | TLoss: {np.mean(t_losses):.4f} | VLoss: {np.mean(v_losses):.4f} | mIoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou; patience_ctr = 0
            torch.save(classifier.state_dict(), f"{exp_dir}/model_best.pth")
            print(f"  ★ New best: {best_iou:.4f}")
        else:
            patience_ctr += 1
            if patience_ctr >= exp['patience']:
                print(f"  ⚠ Early stopping at epoch {epoch+1}"); break

    # Save experiment results
    exp_results[name] = {'best_iou': best_iou, 'history': history, 'exp_dir': exp_dir}

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='Train'); axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title(f'{name} — Loss'); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history['val_iou'], color='coral')
    axes[1].axhline(y=best_iou, color='red', linestyle='--', label=f'Best: {best_iou:.4f}')
    axes[1].set_title('Val mIoU'); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(f"{exp_dir}/curves.png", dpi=150); plt.close()

    torch.cuda.empty_cache()

# ── Comparison table ─────────────────────────────────────────
print("\n" + "="*60)
print("HYPERPARAMETER EXPERIMENT RESULTS")
print(f"{'Experiment':<30} {'Best Val mIoU':>15}")
print("-"*50)
best_exp_name = max(exp_results, key=lambda k: exp_results[k]['best_iou'])
for name, result in sorted(exp_results.items(), key=lambda x: -x[1]['best_iou']):
    flag = " ← WINNER" if name == best_exp_name else ""
    print(f"{name:<30} {result['best_iou']:>14.4f}{flag}")
print("="*60)
print(f"\n→ Best experiment: {best_exp_name}")
print(f"→ Best Val mIoU  : {exp_results[best_exp_name]['best_iou']:.4f}")
print("→ Report these settings to team for final combined model!")

# ============================================================
# CELL 8 — TTA (Test-Time Augmentation)
# ============================================================
class TTAInference:
    """
    Runs 4 TTA variants per image and averages softmax probabilities.
    Gives free IoU improvement with no retraining.
    """
    def __init__(self, classifier, backbone, device, n_classes=10):
        self.classifier = classifier
        self.backbone   = backbone
        self.device     = device
        self.n_classes  = n_classes

    def _forward(self, imgs):
        feats   = self.backbone.forward_features(imgs)['x_norm_patchtokens']
        logits  = self.classifier(feats)
        outputs = F.interpolate(logits, size=imgs.shape[2:], mode='bilinear', align_corners=False)
        return F.softmax(outputs, dim=1)

    @torch.no_grad()
    def predict(self, imgs):
        """imgs: [B, 3, H, W]  → returns [B, H, W] class predictions"""
        imgs = imgs.to(self.device)
        B, C, H, W = imgs.shape
        accumulated = torch.zeros(B, self.n_classes, H, W, device=self.device)

        # Variant 1: Original
        accumulated += self._forward(imgs)

        # Variant 2: Horizontal flip
        flipped = torch.flip(imgs, dims=[-1])
        accumulated += torch.flip(self._forward(flipped), dims=[-1])

        # Variant 3: Scale 0.75× → upsample back
        small = F.interpolate(imgs, scale_factor=0.75, mode='bilinear', align_corners=False)
        pred_small = self._forward(small)
        accumulated += F.interpolate(pred_small, size=(H, W), mode='bilinear', align_corners=False)

        # Variant 4: Scale 1.25× → crop back to original size
        large = F.interpolate(imgs, scale_factor=1.25, mode='bilinear', align_corners=False)
        pred_large = self._forward(large)
        accumulated += F.interpolate(pred_large, size=(H, W), mode='bilinear', align_corners=False)

        accumulated /= 4
        return torch.argmax(accumulated, dim=1)


print("TTA defined ✓")

# ============================================================
# CELL 9 — TEST INFERENCE PIPELINE
# (Run this after loading the best model from any member)
# ============================================================

def run_test_inference(model_path, use_tta=True):
    """
    Full test inference pipeline.
    Loads model → runs on test set → saves masks, colorized masks, metrics.

    Args:
        model_path: path to .pth weights file
        use_tta   : whether to use Test-Time Augmentation
    """
    print(f"\n{'='*60}")
    print(f"INFERENCE PIPELINE")
    print(f"Model     : {model_path}")
    print(f"TTA       : {'ON' if use_tta else 'OFF'}")
    print(f"Test images: {len(test_dataset)}")
    print(f"{'='*60}")

    # ── Load model ───────────────────────────────────────────
    classifier_inf = SegmentationHeadConvNeXt(n_emb, CONFIG['n_classes'], tokenW, tokenH).to(device)
    classifier_inf.load_state_dict(torch.load(model_path, map_location=device))
    classifier_inf.eval()
    backbone.eval()
    print(f"Model loaded from {model_path} ✓")

    tta = TTAInference(classifier_inf, backbone, device)

    # ── Output directories ───────────────────────────────────
    pred_dir   = os.path.join(CONFIG['output_dir'], 'predictions', 'masks')
    color_dir  = os.path.join(CONFIG['output_dir'], 'predictions', 'masks_color')
    comp_dir   = os.path.join(CONFIG['output_dir'], 'predictions', 'comparisons')
    os.makedirs(pred_dir,  exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(comp_dir,  exist_ok=True)

    saved_count = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Running inference", unit="batch")
        for imgs, fnames in pbar:
            if use_tta:
                preds = tta.predict(imgs)          # [B, H, W]
            else:
                imgs_gpu = imgs.to(device)
                feats    = backbone.forward_features(imgs_gpu)['x_norm_patchtokens']
                logits   = classifier_inf(feats)
                outputs  = F.interpolate(logits, size=imgs.shape[2:], mode='bilinear', align_corners=False)
                preds    = torch.argmax(outputs, dim=1)

            for i in range(imgs.shape[0]):
                fname     = fnames[i]
                base_name = os.path.splitext(fname)[0]
                pred_mask = preds[i].cpu().numpy().astype(np.uint8)

                # Raw mask (class IDs 0–9)
                Image.fromarray(pred_mask).save(
                    os.path.join(pred_dir, f"{base_name}_pred.png"))

                # Colorized mask (RGB)
                h_p, w_p = pred_mask.shape
                color_mask = np.zeros((h_p, w_p, 3), dtype=np.uint8)
                for class_id in range(CONFIG['n_classes']):
                    color_mask[pred_mask == class_id] = COLOR_PALETTE[class_id]
                cv2.imwrite(
                    os.path.join(color_dir, f"{base_name}_color.png"),
                    cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))

                # Comparison (input | prediction) for first 10
                if saved_count < 10:
                    img_np = imgs[i].cpu().numpy()
                    mean_v = np.array([0.485, 0.456, 0.406])
                    std_v  = np.array([0.229, 0.224, 0.225])
                    img_np = np.moveaxis(img_np, 0, -1)
                    img_np = np.clip((img_np * std_v + mean_v) * 255, 0, 255).astype(np.uint8)

                    comparison = np.hstack([
                        cv2.resize(img_np, (w_p, h_p)),
                        color_mask
                    ])
                    cv2.imwrite(
                        os.path.join(comp_dir, f"sample_{saved_count:03d}_{base_name}.png"),
                        cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

                saved_count += 1

            pbar.set_postfix(saved=saved_count)

    print(f"\n✅ Inference complete!")
    print(f"   Predictions saved: {saved_count}")
    print(f"   Expected         : {len(test_dataset)}")
    assert saved_count == len(test_dataset), f"❌ Mismatch! Expected {len(test_dataset)}, got {saved_count}"
    print(f"   Count verified   : ✓")
    print(f"\n   Output structure:")
    print(f"   {CONFIG['output_dir']}/predictions/")
    print(f"   ├── masks/          ← {saved_count} raw prediction masks (class IDs 0–9)")
    print(f"   ├── masks_color/    ← {saved_count} colorized RGB masks")
    print(f"   └── comparisons/    ← 10 side-by-side input|prediction images")
    return saved_count


# ── Run inference ─────────────────────────────────────────────
# Change model_path to whichever member's model performed best
# Options:
#   '/kaggle/working/member1_outputs/model_finetuned_best.pth'
#   '/kaggle/working/member2_outputs/model_augmented_best.pth'
#   f"{exp_results[best_exp_name]['exp_dir']}/model_best.pth"

BEST_MODEL_PATH = f"{exp_results[best_exp_name]['exp_dir']}/model_best.pth"
saved = run_test_inference(BEST_MODEL_PATH, use_tta=True)

# ============================================================
# CELL 10 — VALIDATION EVALUATION (with ground truth)
# ============================================================
def evaluate_on_val(model_path, use_tta=False):
    """
    Evaluate model on validation set with known ground truth.
    Gives per-class IoU breakdown.
    """
    classifier_eval = SegmentationHeadConvNeXt(n_emb, CONFIG['n_classes'], tokenW, tokenH).to(device)
    classifier_eval.load_state_dict(torch.load(model_path, map_location=device))
    classifier_eval.eval()
    tta = TTAInference(classifier_eval, backbone, device)

    all_ious, all_class_ious = [], []

    print(f"\nEvaluating on val set (TTA={'ON' if use_tta else 'OFF'})...")
    with torch.no_grad():
        for imgs, masks, _ in tqdm(val_loader, desc="Val eval"):
            imgs, masks = imgs.to(device), masks.to(device)
            if use_tta:
                preds = tta.predict(imgs)
                # Convert predictions to pseudo-logits for compute_iou
                logits = F.one_hot(preds, CONFIG['n_classes']).permute(0,3,1,2).float()
            else:
                feats   = backbone.forward_features(imgs)['x_norm_patchtokens']
                logits  = classifier_eval(feats)
                logits  = F.interpolate(logits, size=imgs.shape[2:], mode='bilinear', align_corners=False)

            iou, iou_list = compute_iou(logits, masks)
            all_ious.append(iou); all_class_ious.append(iou_list)

    mean_iou = np.nanmean(all_ious)
    avg_class = np.nanmean(all_class_ious, axis=0)

    print(f"\n{'='*50}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"\nPer-class IoU:")
    for i, (name, iou) in enumerate(zip(CLASS_NAMES, avg_class)):
        bar  = '█' * int((iou if not np.isnan(iou) else 0) * 20)
        s    = f"{iou:.4f}" if not np.isnan(iou) else "  N/A"
        print(f"  [{i}] {name:<16}: {s}  {bar}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ['red' if iou < 0.3 else 'orange' if iou < 0.5 else 'green'
                for iou in [0 if np.isnan(v) else v for v in avg_class]]
    ax.bar(CLASS_NAMES, [0 if np.isnan(v) else v for v in avg_class], color=colors, edgecolor='black')
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.axhline(y=mean_iou, color='red', linestyle='--', label=f'Mean: {mean_iou:.4f}')
    ax.set_title('Per-Class IoU'); ax.set_ylabel('IoU'); ax.set_ylim(0, 1)
    ax.legend(); ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/per_class_iou.png", dpi=150); plt.close()
    print(f"\n✓ Chart saved: {CONFIG['output_dir']}/per_class_iou.png")

    return mean_iou, avg_class

# Run val evaluation on best model
val_iou_no_tta, _ = evaluate_on_val(BEST_MODEL_PATH, use_tta=False)
val_iou_tta, _    = evaluate_on_val(BEST_MODEL_PATH, use_tta=True)

print(f"\n📊 TTA COMPARISON:")
print(f"   Without TTA : {val_iou_no_tta:.4f}")
print(f"   With TTA    : {val_iou_tta:.4f}")
print(f"   TTA boost   : +{val_iou_tta - val_iou_no_tta:.4f}")

# ============================================================
# CELL 11 — FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("✅ MEMBER 3 COMPLETE")
print(f"\nHyperparameter Results:")
for name, result in sorted(exp_results.items(), key=lambda x: -x[1]['best_iou']):
    print(f"  {name}: {result['best_iou']:.4f}")
print(f"\nBest experiment    : {best_exp_name}")
print(f"Val mIoU (no TTA)  : {val_iou_no_tta:.4f}")
print(f"Val mIoU (with TTA): {val_iou_tta:.4f}")
print(f"Test predictions   : {saved} files saved")
print(f"\nReport to team:")
print(f"  → Best LR        : {[e for e in EXPERIMENTS if e['name']==best_exp_name][0]['lr']}")
print(f"  → Best scheduler : {[e for e in EXPERIMENTS if e['name']==best_exp_name][0]['scheduler']}")
print(f"  → TTA boost      : +{val_iou_tta - val_iou_no_tta:.4f}")
print("="*60)
