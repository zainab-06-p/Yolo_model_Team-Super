# ╔══════════════════════════════════════════════════════════════════════╗
# ║  MEMBER 2 — Augmentation Pipeline + Class Weights                   ║
# ║  Strategy: Frozen backbone + heavy augmentation + weighted loss      ║
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
# CELL 2 — INSTALL DEPENDENCIES
# ============================================================
# !pip install albumentations -q
import albumentations as A
from albumentations.pytorch import ToTensorV2
print(f"Albumentations version: {A.__version__} ✓")

# ============================================================
# CELL 3 — IMPORTS
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

import os, cv2, random
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
# CELL 4 — CONFIGURATION
# ============================================================
CONFIG = {
    # KAGGLE PATHS (your uploaded dataset)
    'train_dir' : '/kaggle/input/datasets/adiinamdar/yolo-training-data/Offroad_Segmentation_Training_Dataset/train',
    'val_dir'   : '/kaggle/input/datasets/adiinamdar/yolo-training-data/Offroad_Segmentation_Training_Dataset/val',
    'output_dir': '/kaggle/working/member2_outputs',

    'img_w': 462,
    'img_h': 252,

    'n_epochs'   : 50,
    'batch_size' : 8,
    'lr'         : 1e-4,
    'weight_decay': 0.01,
    'gradient_clip': 1.0,
    'early_stop_patience': 15,
    'n_classes'  : 10,
    'num_workers': 2,
    'checkpoint_every': 5,
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(f"{CONFIG['output_dir']}/checkpoints", exist_ok=True)

# ============================================================
# CELL 5 — CLASS MAPPING
# ============================================================
VALUE_MAP = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

RARE_CLASSES = [5, 6]  # Ground Clutter, Logs

# ============================================================
# CELL 6 — DATASET AUDIT (run this FIRST — outputs class weights)
# ============================================================
def audit_dataset(data_dir, n_classes=10, sample_limit=500):
    """
    Scans training masks and computes pixel-frequency per class.
    Run this once to understand class imbalance.
    """
    mask_dir = os.path.join(data_dir, 'Segmentation')
    files    = sorted(os.listdir(mask_dir))[:sample_limit]

    print(f"Auditing {len(files)} masks...")
    pixel_counts = np.zeros(n_classes, dtype=np.int64)

    for fname in tqdm(files, desc="Auditing"):
        mask = np.array(Image.open(os.path.join(mask_dir, fname)))
        for raw_val, class_id in VALUE_MAP.items():
            pixel_counts[class_id] += (mask == raw_val).sum()

    total = pixel_counts.sum()
    print("\n📊 CLASS DISTRIBUTION:")
    print(f"{'Class':<20} {'Pixels':>12} {'Frequency':>10} {'Weight':>8}")
    print("-" * 55)

    freqs   = pixel_counts / total
    weights = 1.0 / np.sqrt(freqs + 1e-4)
    weights = weights / weights.sum() * n_classes

    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<20} {pixel_counts[i]:>12,} {freqs[i]*100:>9.3f}%  {weights[i]:>7.2f}x")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['red' if f < 0.01 else 'orange' if f < 0.05 else 'steelblue' for f in freqs]
    axes[0].bar(CLASS_NAMES, freqs * 100, color=colors)
    axes[0].set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    axes[0].set_title('Class Pixel Frequency (%)'); axes[0].set_ylabel('%')
    axes[0].grid(axis='y')

    axes[1].bar(CLASS_NAMES, weights, color='purple', alpha=0.7)
    axes[1].set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    axes[1].set_title('Computed Class Weights'); axes[1].set_ylabel('Weight')
    axes[1].grid(axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'class_distribution.png'), dpi=150)
    plt.close()
    print(f"\n✓ Chart saved to {CONFIG['output_dir']}/class_distribution.png")

    return torch.tensor(weights, dtype=torch.float32)


# Run audit
class_weights = audit_dataset(CONFIG['train_dir'])
class_weights  = class_weights.to(device)
print(f"\nClass weights tensor: {class_weights}")
print("→ Share class_weights with Member 1 via team chat!")

# ============================================================
# CELL 7 — AUGMENTATION PIPELINE (Albumentations)
# ============================================================
h, w = CONFIG['img_h'], CONFIG['img_w']

train_transform = A.Compose([
    # ── Tier 1: Geometric ────────────────────────────────────
    A.HorizontalFlip(p=0.5),
    # NO vertical flip — sky must stay on top
    A.ShiftScaleRotate(
        shift_limit=0.1, scale_limit=0.3,
        rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.7
    ),
    A.RandomResizedCrop(
        size=(h, w),  # Use size tuple instead of height/width
        scale=(0.5, 1.0),  # Scale must be <= 1.0 in new API
        ratio=(0.75, 1.33),
        interpolation=cv2.INTER_LINEAR, p=1.0
    ),
    A.Perspective(scale=(0.02, 0.05), keep_size=True, p=0.2),

    # ── Tier 2: Photometric ──────────────────────────────────
    A.OneOf([
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        A.RandomGamma(gamma_limit=(70, 130)),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),
    ], p=0.5),

    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7)),
        A.MotionBlur(blur_limit=7),
    ], p=0.2),

    A.GaussNoise(var_limit=(10, 50), mean=0, p=0.15),
    A.RandomShadow(
        shadow_roi=(0, 0.5, 1, 1),
        num_shadows_lower=1, num_shadows_upper=2,
        shadow_dimension=5, p=0.2
    ),
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.25, alpha_coef=0.1, p=0.1),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),

    # ── Tier 3: Occlusion ────────────────────────────────────
    A.CoarseDropout(
        max_holes=6, max_height=40, max_width=40,
        min_holes=1, min_height=10, min_width=10,
        fill_value=0, mask_fill_value=255, p=0.15
    ),
    A.ToGray(p=0.05),

    # ── Normalize ────────────────────────────────────────────
    A.Resize(height=h, width=w, p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})


val_transform = A.Compose([
    A.Resize(height=h, width=w),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})


# ── Visual verification of augmentation ─────────────────────
def verify_augmentation(dataset, n=5, save_dir=None):
    print("Verifying augmentation pipeline...")
    fig, axes = plt.subplots(n, 2, figsize=(10, n * 3))
    for i in range(n):
        img_t, mask_t = dataset[i]
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img_disp = ((img_t * std + mean) * 255).clamp(0, 255).permute(1,2,0).numpy().astype(np.uint8)
        axes[i, 0].imshow(img_disp); axes[i, 0].set_title(f"Image {i}")
        axes[i, 1].imshow(mask_t.numpy(), vmin=0, vmax=9, cmap='tab10')
        axes[i, 1].set_title(f"Mask {i} | unique={mask_t.unique().tolist()}")
    plt.tight_layout()
    path = os.path.join(save_dir or CONFIG['output_dir'], 'augmentation_check.png')
    plt.savefig(path, dpi=100); plt.close()
    print(f"✓ Augmentation check saved: {path}")

print("Augmentation pipeline defined ✓")

# ============================================================
# CELL 8 — DATASET WITH ALBUMENTATIONS
# ============================================================
def convert_mask_np(mask_pil):
    """Convert 16-bit PIL mask → numpy uint8 class IDs."""
    arr = np.array(mask_pil)
    out = np.zeros_like(arr, dtype=np.uint8)
    for raw_val, class_id in VALUE_MAP.items():
        out[arr == raw_val] = class_id
    return out


class AugmentedDesertDataset(Dataset):
    def __init__(self, data_dir, transform=None, build_sampler_weights=False):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.mask_dir  = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.filenames = sorted(os.listdir(self.image_dir))
        self.sampler_weights = None

        if build_sampler_weights:
            self._build_sampler_weights()

    def _build_sampler_weights(self):
        """Give 3x weight to images containing rare classes."""
        print("Building sampler weights (scanning for rare classes)...")
        weights = []
        for fname in tqdm(self.filenames, desc="Scanning masks", leave=False):
            mask = np.array(Image.open(os.path.join(self.mask_dir, fname)))
            has_rare = any((mask == raw_val).any()
                          for raw_val, class_id in VALUE_MAP.items()
                          if class_id in RARE_CLASSES)
            weights.append(3.0 if has_rare else 1.0)
        self.sampler_weights = weights
        rare_count = sum(1 for w in weights if w > 1.0)
        print(f"Images with rare classes: {rare_count}/{len(self.filenames)}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img   = np.array(Image.open(os.path.join(self.image_dir, fname)).convert('RGB'))
        mask  = convert_mask_np(Image.open(os.path.join(self.mask_dir, fname)))

        if self.transform:
            result = self.transform(image=img, mask=mask)
            img    = result['image']   # [3, H, W] float tensor
            mask   = torch.tensor(result['mask'], dtype=torch.long)

        return img, mask


# Build datasets
train_dataset = AugmentedDesertDataset(
    CONFIG['train_dir'],
    transform=train_transform,
    build_sampler_weights=True
)
val_dataset = AugmentedDesertDataset(CONFIG['val_dir'], transform=val_transform)

# WeightedRandomSampler — oversample images with rare classes
sampler = WeightedRandomSampler(
    weights=train_dataset.sampler_weights,
    num_samples=len(train_dataset),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size  = CONFIG['batch_size'],
    sampler     = sampler,         # use weighted sampler
    num_workers = CONFIG['num_workers'],
    pin_memory  = True,
    persistent_workers=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size  = CONFIG['batch_size'],
    shuffle     = False,
    num_workers = CONFIG['num_workers'],
    pin_memory  = True,
    persistent_workers=True
)

print(f"Train samples : {len(train_dataset)}")
print(f"Val samples   : {len(val_dataset)}")

# ── Sanity check ─────────────────────────────────────────────
imgs, masks = next(iter(train_loader))
print(f"\nBatch: imgs={imgs.shape}, masks={masks.shape}, dtype={masks.dtype}")
print(f"Mask unique values: {masks.unique().tolist()}")
assert masks.max() <= 9, "❌ Mask values out of range!"
print("Dataset sanity check PASSED ✓")

# ── Verify augmentation visually ─────────────────────────────
verify_augmentation(train_dataset)

# ============================================================
# CELL 9 — COPY-PASTE FOR RARE CLASSES
# ============================================================
class CopyPasteAugmentor:
    """
    Builds a pool of rare-class crops from training data.
    Pastes them into random training images to oversample rare classes.
    """
    def __init__(self, dataset, rare_class_ids=RARE_CLASSES, pool_size=200):
        self.rare_class_ids = rare_class_ids
        self.pool = []
        self._build_pool(dataset, pool_size)

    def _build_pool(self, dataset, pool_size):
        print(f"Building copy-paste pool (target: {pool_size} crops)...")
        for idx in tqdm(range(len(dataset)), desc="Scanning", leave=False):
            if len(self.pool) >= pool_size:
                break
            fname = dataset.filenames[idx]
            img   = np.array(Image.open(os.path.join(dataset.image_dir, fname)).convert('RGB'))
            mask  = convert_mask_np(Image.open(os.path.join(dataset.mask_dir, fname)))

            for class_id in self.rare_class_ids:
                binary = (mask == class_id).astype(np.uint8)
                if binary.sum() < 100:
                    continue
                num_labels, labels = cv2.connectedComponents(binary)
                for label_id in range(1, num_labels):
                    component = (labels == label_id)
                    if component.sum() < 100:
                        continue
                    rows = np.where(np.any(component, axis=1))[0]
                    cols = np.where(np.any(component, axis=0))[0]
                    r0, r1, c0, c1 = rows[0], rows[-1]+1, cols[0], cols[-1]+1
                    self.pool.append({
                        'img_crop' : img[r0:r1, c0:c1].copy(),
                        'mask_crop': mask[r0:r1, c0:c1].copy(),
                        'binary'   : component[r0:r1, c0:c1],
                        'class_id' : class_id,
                    })
        print(f"Pool built: {len(self.pool)} crops ✓")

    def apply(self, image: np.ndarray, mask: np.ndarray, n_pastes: int = 2):
        """Paste n_pastes crops into image/mask pair."""
        if not self.pool:
            return image, mask
        H, W = image.shape[:2]
        result_img, result_mask = image.copy(), mask.copy()

        for _ in range(n_pastes):
            crop = random.choice(self.pool)
            ih, iw = crop['img_crop'].shape[:2]
            if ih >= H or iw >= W:
                continue
            py = random.randint(0, H - ih)
            px = random.randint(0, W - iw)
            b  = crop['binary']
            result_img[py:py+ih, px:px+iw][b] = crop['img_crop'][b]
            result_mask[py:py+ih, px:px+iw][b] = crop['mask_crop'][b]

        return result_img, result_mask


# Build copy-paste pool
cp_augmentor = CopyPasteAugmentor(train_dataset, pool_size=200)
print("Copy-paste augmentor ready ✓")

# ============================================================
# CELL 10 — MODEL (same architecture as Member 1 — MUST MATCH)
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


# Loss functions
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__(); self.smooth = smooth
    def forward(self, logits, targets):
        nc = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        oh    = F.one_hot(targets, nc).permute(0,3,1,2).float()
        inter = (probs * oh).sum(dim=(2,3))
        union = probs.sum(dim=(2,3)) + oh.sum(dim=(2,3))
        return 1 - ((2*inter + self.smooth)/(union + self.smooth)).mean()


def compute_iou(logits, targets, n_classes=10):
    preds   = torch.argmax(logits, dim=1).view(-1)
    targets = targets.view(-1)
    ious = []
    for c in range(n_classes):
        pc = preds == c; tc = targets == c
        inter = (pc & tc).sum().float()
        union = (pc | tc).sum().float()
        ious.append(float('nan') if union == 0 else (inter/union).item())
    return np.nanmean(ious), ious


print("Model and loss defined ✓")

# ============================================================
# CELL 11 — LOAD BACKBONE (frozen)
# ============================================================
print("Loading DINOv2 (frozen)...")
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
backbone = backbone.to(device).eval()
for param in backbone.parameters():
    param.requires_grad = False

with torch.no_grad():
    sample = torch.zeros(1, 3, h, w).to(device)
    feats  = backbone.forward_features(sample)['x_norm_patchtokens']
    n_embedding = feats.shape[2]
    tokenH, tokenW = h // 14, w // 14

classifier = SegmentationHeadConvNeXt(n_embedding, CONFIG['n_classes'], tokenW, tokenH).to(device)
print(f"Backbone: frozen | Head params: {sum(p.numel() for p in classifier.parameters()):,} ✓")

# ============================================================
# CELL 12 — TRAINING LOOP
# ============================================================
optimizer = optim.AdamW(classifier.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['n_epochs'], eta_min=1e-6)
scaler    = GradScaler()

ce_loss   = nn.CrossEntropyLoss(weight=class_weights)
dice_loss = DiceLoss()

history = {k: [] for k in ['train_loss','val_loss','train_iou','val_iou','lr']}
best_val_iou    = 0.0
patience_counter= 0

print(f"\n{'='*60}")
print("MEMBER 2 TRAINING: Augmentation + Class Weights (Frozen Backbone)")
print(f"Epochs: {CONFIG['n_epochs']} | Batch: {CONFIG['batch_size']} | LR: {CONFIG['lr']}")
print(f"{'='*60}")

for epoch in range(CONFIG['n_epochs']):
    # ── Training ─────────────────────────────────────────────
    classifier.train()
    train_losses = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['n_epochs']} [Train]", leave=False)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)

        with torch.no_grad():
            feats = backbone.forward_features(imgs)['x_norm_patchtokens']

        optimizer.zero_grad()
        with autocast():
            logits  = classifier(feats)
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode='bilinear', align_corners=False)
            loss    = 0.5 * ce_loss(outputs, masks) + 0.5 * dice_loss(outputs, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), CONFIG['gradient_clip'])
        scaler.step(optimizer); scaler.update()

        train_losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()

    # ── Validation ───────────────────────────────────────────
    classifier.eval()
    val_losses, val_ious = [], []
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            feats   = backbone.forward_features(imgs)['x_norm_patchtokens']
            logits  = classifier(feats)
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode='bilinear', align_corners=False)
            loss    = ce_loss(outputs, masks)
            iou, iou_list = compute_iou(outputs, masks)
            val_losses.append(loss.item()); val_ious.append(iou)

    epoch_val_iou = np.nanmean(val_ious)
    history['train_loss'].append(np.mean(train_losses))
    history['val_loss'].append(np.mean(val_losses))
    history['train_iou'].append(0.0)
    history['val_iou'].append(float(epoch_val_iou))
    history['lr'].append(optimizer.param_groups[0]['lr'])

    print(f"\nEpoch {epoch+1:02d}/{CONFIG['n_epochs']} | "
          f"Train Loss: {np.mean(train_losses):.4f} | "
          f"Val Loss: {np.mean(val_losses):.4f} | "
          f"Val mIoU: {epoch_val_iou:.4f}")

    if epoch_val_iou > best_val_iou:
        best_val_iou = epoch_val_iou
        patience_counter = 0
        torch.save(classifier.state_dict(), f"{CONFIG['output_dir']}/model_augmented_best.pth")
        print(f"  ★ New best: {best_val_iou:.4f} → saved model_augmented_best.pth")
        # Print per-class IoU on improvement
        print("  Per-class IoU:")
        for i, (name, iou) in enumerate(zip(CLASS_NAMES, iou_list)):
            s = f"{iou:.4f}" if not np.isnan(iou) else " N/A"
            flag = " ← RARE" if i in RARE_CLASSES else ""
            print(f"    [{i}] {name:<16}: {s}{flag}")
    else:
        patience_counter += 1
        if patience_counter >= CONFIG['early_stop_patience']:
            print(f"\n⚠ Early stopping at epoch {epoch+1}")
            break

    if (epoch+1) % CONFIG['checkpoint_every'] == 0:
        torch.save(classifier.state_dict(),
                   f"{CONFIG['output_dir']}/checkpoints/aug_epoch{epoch+1}.pth")

    if (epoch+1) % 10 == 0:
        torch.cuda.empty_cache()

# ============================================================
# CELL 13 — SAVE PLOTS & REPORT RESULT
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history['train_loss'], label='Train'); axes[0].plot(history['val_loss'], label='Val')
axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True)
axes[1].plot(history['val_iou'], color='coral')
axes[1].set_title('Val mIoU'); axes[1].grid(True)
axes[1].axhline(y=best_val_iou, color='red', linestyle='--', label=f'Best: {best_val_iou:.4f}')
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/training_curves_aug.png", dpi=150)
plt.close()
print(f"✓ Curves saved")

print("\n" + "="*60)
print("✅ MEMBER 2 TRAINING COMPLETE")
print(f"   Strategy   : Augmentation + Class Weights (Frozen backbone)")
print(f"   Best mIoU  : {best_val_iou:.4f}")
print(f"   Model file : model_augmented_best.pth")
print("   → Report this number to team for ablation table!")
print("="*60)
