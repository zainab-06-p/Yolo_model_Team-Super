# ╔══════════════════════════════════════════════════════════════════════╗
# ║  MEMBER 1 — DINOv2 Fine-Tuning + Core Training Pipeline             ║
# ║  Strategy: 2-Phase Training — Frozen → Partial Unfreeze             ║
# ║  Duality AI Hackathon | Kaggle T4 GPU                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ============================================================
# CELL 1 — GPU CHECK (Run this first, confirm T4 is active)
# ============================================================
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout)

import torch
print(f"\nPyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# CELL 2 — INSTALL DEPENDENCIES
# ============================================================
# !pip install albumentations -q
# albumentations already available on Kaggle, uncomment only if needed

# ============================================================
# CELL 3 — IMPORTS
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

import os
import cv2
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')

print("All imports successful ✓")

# ============================================================
# CELL 4 — CONFIGURATION  (edit paths here only)
# ============================================================
CONFIG = {
    # ── Paths (update dataset name to match yours) ──────────
    'train_dir' : '/kaggle/input/YOUR-DATASET/Offroad_Segmentation_Training_Dataset/train',
    'val_dir'   : '/kaggle/input/YOUR-DATASET/Offroad_Segmentation_Training_Dataset/val',
    'test_dir'  : '/kaggle/input/YOUR-DATASET/Offroad_Segmentation_testImages',
    'output_dir': '/kaggle/working/member1_outputs',

    # ── Image size (must be multiples of 14 for DINOv2) ─────
    'img_w': 462,   # 33 × 14
    'img_h': 252,   # 18 × 14

    # ── Phase 1 (frozen backbone) ────────────────────────────
    'phase1_epochs'    : 25,
    'phase1_batch_size': 8,
    'phase1_lr'        : 3e-4,

    # ── Phase 2 (unfreeze last 4 blocks) ────────────────────
    'phase2_epochs'      : 25,
    'phase2_batch_size'  : 4,
    'phase2_head_lr'     : 3e-5,
    'phase2_backbone_lr' : 3e-6,

    # ── General ──────────────────────────────────────────────
    'n_classes'        : 10,
    'weight_decay'     : 0.01,
    'gradient_clip'    : 1.0,
    'early_stop_patience': 12,
    'checkpoint_every' : 5,
    'seed'             : 42,
    'num_workers'      : 2,
}

# Set seeds for reproducibility
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(f"{CONFIG['output_dir']}/checkpoints", exist_ok=True)
print(f"Output dir: {CONFIG['output_dir']}")

# ============================================================
# CELL 5 — CLASS MAPPING
# ============================================================
VALUE_MAP = {
    0    : 0,   # Background
    100  : 1,   # Trees
    200  : 2,   # Lush Bushes
    300  : 3,   # Dry Grass
    500  : 4,   # Dry Bushes
    550  : 5,   # Ground Clutter
    700  : 6,   # Logs
    800  : 7,   # Rocks
    7100 : 8,   # Landscape
    10000: 9,   # Sky
}

CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

# Class weights — rare classes get higher weight
# Logs (idx 6) = 0.07% of pixels → highest weight
CLASS_WEIGHTS = torch.tensor([
    1.0,  # Background
    2.0,  # Trees
    2.0,  # Lush Bushes
    1.5,  # Dry Grass
    2.0,  # Dry Bushes
    3.0,  # Ground Clutter
    8.0,  # Logs           ← rarest: 0.07%
    3.0,  # Rocks
    1.0,  # Landscape
    1.0,  # Sky
]).to(device)

print("Class mapping defined ✓")
print(f"Classes: {CLASS_NAMES}")

# ============================================================
# CELL 6 — DATASET
# ============================================================
def convert_mask(mask_pil):
    """Convert 16-bit raw mask values → contiguous class IDs 0–9."""
    arr = np.array(mask_pil)
    out = np.zeros_like(arr, dtype=np.uint8)
    for raw_val, class_id in VALUE_MAP.items():
        out[arr == raw_val] = class_id
    return Image.fromarray(out)


class DesertDataset(Dataset):
    def __init__(self, data_dir, img_transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.mask_dir  = os.path.join(data_dir, 'Segmentation')
        self.img_transform  = img_transform
        self.mask_transform = mask_transform
        self.filenames = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img  = Image.open(os.path.join(self.image_dir, fname)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir,  fname))
        mask = convert_mask(mask)

        if self.img_transform:
            img = self.img_transform(img)
        if self.mask_transform:
            # mask_transform returns float tensor — multiply by 255 to restore 0-9 int values
            mask = self.mask_transform(mask)
            mask = (mask * 255).long().squeeze(0)   # [H, W] long

        return img, mask


h, w = CONFIG['img_h'], CONFIG['img_w']

img_transform = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

train_dataset = DesertDataset(CONFIG['train_dir'], img_transform, mask_transform)
val_dataset   = DesertDataset(CONFIG['val_dir'],   img_transform, mask_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size  = CONFIG['phase1_batch_size'],
    shuffle     = True,
    num_workers = CONFIG['num_workers'],
    pin_memory  = True,
    persistent_workers=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size  = CONFIG['phase1_batch_size'],
    shuffle     = False,
    num_workers = CONFIG['num_workers'],
    pin_memory  = True,
    persistent_workers=True
)

print(f"Train samples : {len(train_dataset)}")
print(f"Val samples   : {len(val_dataset)}")

# ── Quick sanity check ──
imgs, masks = next(iter(train_loader))
print(f"\nBatch shapes  : imgs={imgs.shape}, masks={masks.shape}")
print(f"Mask dtype    : {masks.dtype}")
print(f"Mask unique   : {masks.unique().tolist()}")
assert masks.max() <= 9, "❌ Mask values out of range — check VALUE_MAP!"
print("Dataset sanity check PASSED ✓")

# ============================================================
# CELL 7 — MODEL ARCHITECTURE
# ============================================================
class SegmentationHeadConvNeXt(nn.Module):
    """
    ConvNeXt-style decoder head on top of DINOv2 patch tokens.
    Added: extra block + dropout for regularization.
    """
    def __init__(self, in_channels, out_channels, tokenW, tokenH, dropout=0.1):
        super().__init__()
        self.tokenH = tokenH
        self.tokenW = tokenW

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=7, padding=3),
            nn.GELU(),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.dropout    = nn.Dropout2d(p=dropout)
        self.classifier = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.tokenH, self.tokenW, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.dropout(x)
        return self.classifier(x)


# ============================================================
# CELL 8 — LOSS FUNCTIONS
# ============================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union        = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice         = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt      = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


def get_phased_loss(epoch, total_phase1=25, total_phase2=25):
    """
    Returns the loss function appropriate for the current epoch.
    Epoch 1–15:  0.5 × WeightedCE  + 0.5 × Dice
    Epoch 15–35: 0.3 × WeightedCE  + 0.7 × Dice
    Epoch 35–50: 0.2 × FocalLoss   + 0.8 × Dice
    """
    ce_loss    = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    dice_loss  = DiceLoss()
    focal_loss = FocalLoss(gamma=2.0, weight=CLASS_WEIGHTS)

    if epoch < 15:
        def loss_fn(logits, targets):
            return 0.5 * ce_loss(logits, targets) + 0.5 * dice_loss(logits, targets)
    elif epoch < 35:
        def loss_fn(logits, targets):
            return 0.3 * ce_loss(logits, targets) + 0.7 * dice_loss(logits, targets)
    else:
        def loss_fn(logits, targets):
            return 0.2 * focal_loss(logits, targets) + 0.8 * dice_loss(logits, targets)

    return loss_fn


print("Loss functions defined ✓")

# ============================================================
# CELL 9 — METRICS
# ============================================================
def compute_iou(logits, targets, n_classes=10):
    preds = torch.argmax(logits, dim=1).view(-1)
    targets = targets.view(-1)
    iou_list = []
    for c in range(n_classes):
        pred_c   = preds == c
        target_c = targets == c
        inter    = (pred_c & target_c).sum().float()
        union    = (pred_c | target_c).sum().float()
        if union == 0:
            iou_list.append(float('nan'))
        else:
            iou_list.append((inter / union).item())
    return np.nanmean(iou_list), iou_list


def compute_per_class_iou_report(iou_list):
    print("\n  Per-class IoU:")
    for i, (name, iou) in enumerate(zip(CLASS_NAMES, iou_list)):
        bar = '█' * int((iou if not np.isnan(iou) else 0) * 20)
        iou_str = f"{iou:.4f}" if not np.isnan(iou) else "  N/A"
        print(f"    [{i}] {name:<16}: {iou_str}  {bar}")


print("Metrics defined ✓")

# ============================================================
# CELL 10 — LOAD BACKBONE
# ============================================================
print("Loading DINOv2 backbone (dinov2_vits14)...")
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
backbone = backbone.to(device)
backbone.eval()

# PHASE 1: Freeze ALL backbone parameters
for param in backbone.parameters():
    param.requires_grad = False

print("Backbone loaded and FROZEN ✓")

# Get embedding dimension
with torch.no_grad():
    sample = torch.zeros(1, 3, h, w).to(device)
    feats  = backbone.forward_features(sample)['x_norm_patchtokens']
    n_embedding = feats.shape[2]
    tokenH = h // 14
    tokenW = w // 14
    print(f"Patch token shape : {feats.shape}")
    print(f"Embedding dim     : {n_embedding}")
    print(f"Token grid        : {tokenH} × {tokenW}")

# ============================================================
# CELL 11 — CREATE SEGMENTATION HEAD
# ============================================================
classifier = SegmentationHeadConvNeXt(
    in_channels = n_embedding,
    out_channels= CONFIG['n_classes'],
    tokenW      = tokenW,
    tokenH      = tokenH,
    dropout     = 0.1
).to(device)

total_params = sum(p.numel() for p in classifier.parameters())
print(f"Segmentation head params: {total_params:,}")

# ============================================================
# CELL 12 — TRAINING UTILITIES
# ============================================================
def save_checkpoint(state, path):
    torch.save(state, path)
    print(f"  ✓ Checkpoint saved: {path}")


def load_checkpoint(path, classifier, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location=device)
    classifier.load_state_dict(ckpt['classifier_state'])
    if optimizer and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    if scheduler and 'scheduler_state' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state'])
    print(f"  ✓ Checkpoint loaded from: {path}")
    return ckpt.get('epoch', 0), ckpt.get('best_val_iou', 0.0)


def save_training_plots(history, output_dir, tag=''):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history['train_loss'], label='Train', color='steelblue')
    axes[0, 0].plot(history['val_loss'],   label='Val',   color='coral')
    axes[0, 0].set_title('Loss'); axes[0, 0].legend(); axes[0, 0].grid(True)

    axes[0, 1].plot(history['train_iou'], label='Train', color='steelblue')
    axes[0, 1].plot(history['val_iou'],   label='Val',   color='coral')
    axes[0, 1].set_title('mIoU'); axes[0, 1].legend(); axes[0, 1].grid(True)

    axes[1, 0].plot(history['train_dice'], label='Train', color='steelblue')
    axes[1, 0].plot(history['val_dice'],   label='Val',   color='coral')
    axes[1, 0].set_title('Dice'); axes[1, 0].legend(); axes[1, 0].grid(True)

    axes[1, 1].plot(history['lr'], color='green')
    axes[1, 1].set_title('Learning Rate'); axes[1, 1].grid(True)

    plt.suptitle(f'Training Curves {tag}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'training_curves{tag}.png'), dpi=150)
    plt.close()
    print(f"  ✓ Plots saved to {output_dir}/training_curves{tag}.png")


# ============================================================
# CELL 13 — PHASE 1 TRAINING (Frozen backbone, 25 epochs)
# ============================================================
print("\n" + "="*60)
print("PHASE 1: Training with FROZEN backbone")
print(f"Epochs: {CONFIG['phase1_epochs']}  |  Batch: {CONFIG['phase1_batch_size']}  |  LR: {CONFIG['phase1_lr']}")
print("="*60)

optimizer_p1 = optim.AdamW(
    classifier.parameters(),
    lr=CONFIG['phase1_lr'],
    weight_decay=CONFIG['weight_decay']
)

# Warmup + CosineAnnealing
scheduler_p1 = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer_p1, T_0=8, T_mult=2, eta_min=1e-6
)

scaler = GradScaler()  # AMP scaler for FP16

history_p1 = {
    'train_loss': [], 'val_loss': [],
    'train_iou':  [], 'val_iou':  [],
    'train_dice': [], 'val_dice': [],
    'lr': []
}

best_val_iou_p1  = 0.0
patience_counter = 0

for epoch in range(CONFIG['phase1_epochs']):
    # ── Training ─────────────────────────────────────────────
    classifier.train()
    backbone.eval()
    train_losses = []
    loss_fn = get_phased_loss(epoch)

    pbar = tqdm(train_loader, desc=f"P1 Epoch {epoch+1}/{CONFIG['phase1_epochs']} [Train]", leave=False)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)

        with torch.no_grad():
            feats = backbone.forward_features(imgs)['x_norm_patchtokens']

        optimizer_p1.zero_grad()
        with autocast():
            logits  = classifier(feats)
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode='bilinear', align_corners=False)
            loss    = loss_fn(outputs, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer_p1)
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), CONFIG['gradient_clip'])
        scaler.step(optimizer_p1)
        scaler.update()

        train_losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    scheduler_p1.step()

    # ── Validation ───────────────────────────────────────────
    classifier.eval()
    val_losses, val_ious, val_dices = [], [], []

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc=f"P1 Epoch {epoch+1} [Val]", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            feats   = backbone.forward_features(imgs)['x_norm_patchtokens']
            logits  = classifier(feats)
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode='bilinear', align_corners=False)
            loss    = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)(outputs, masks)
            iou, _  = compute_iou(outputs, masks)
            val_losses.append(loss.item())
            val_ious.append(iou)

    epoch_val_iou  = np.nanmean(val_ious)
    epoch_train_iou = np.nanmean([compute_iou(
        F.interpolate(classifier(backbone.forward_features(imgs.to(device))['x_norm_patchtokens']),
                      size=(h, w), mode='bilinear', align_corners=False),
        masks.to(device))[0]
        for imgs, masks in list(val_loader)[:5]
    ])

    history_p1['train_loss'].append(np.mean(train_losses))
    history_p1['val_loss'].append(np.mean(val_losses))
    history_p1['train_iou'].append(float(epoch_train_iou))
    history_p1['val_iou'].append(float(epoch_val_iou))
    history_p1['train_dice'].append(0.0)  # placeholder
    history_p1['val_dice'].append(0.0)
    history_p1['lr'].append(optimizer_p1.param_groups[0]['lr'])

    print(f"\nEpoch {epoch+1:02d}/{CONFIG['phase1_epochs']} | "
          f"Train Loss: {np.mean(train_losses):.4f} | "
          f"Val Loss: {np.mean(val_losses):.4f} | "
          f"Val mIoU: {epoch_val_iou:.4f}")

    # ── Best model & early stopping ──────────────────────────
    if epoch_val_iou > best_val_iou_p1:
        best_val_iou_p1 = epoch_val_iou
        patience_counter = 0
        save_checkpoint({
            'epoch': epoch,
            'classifier_state': classifier.state_dict(),
            'optimizer_state': optimizer_p1.state_dict(),
            'best_val_iou': best_val_iou_p1
        }, f"{CONFIG['output_dir']}/checkpoints/phase1_best.pth")
        print(f"  ★ New best Val mIoU: {best_val_iou_p1:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= CONFIG['early_stop_patience']:
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            break

    # ── Periodic checkpoint ──────────────────────────────────
    if (epoch + 1) % CONFIG['checkpoint_every'] == 0:
        save_checkpoint({
            'epoch': epoch,
            'classifier_state': classifier.state_dict(),
        }, f"{CONFIG['output_dir']}/checkpoints/phase1_epoch{epoch+1}.pth")

    # ── Flush GPU memory ─────────────────────────────────────
    if (epoch + 1) % 10 == 0:
        torch.cuda.empty_cache()

save_training_plots(history_p1, CONFIG['output_dir'], tag='_phase1')
print(f"\n✅ Phase 1 Complete! Best Val mIoU: {best_val_iou_p1:.4f}")
print(f"   Target was > 0.40 — {'PASSED ✓' if best_val_iou_p1 > 0.40 else 'Below target ⚠ — debug before Phase 2'}")

# ============================================================
# CELL 14 — PHASE 2: UNFREEZE LAST 4 BACKBONE BLOCKS
# ============================================================
# Load best Phase 1 weights before unfreezing
load_checkpoint(
    f"{CONFIG['output_dir']}/checkpoints/phase1_best.pth",
    classifier
)

# Unfreeze last 4 transformer blocks of DINOv2
for block in backbone.blocks[-4:]:
    for param in block.parameters():
        param.requires_grad = True

unfrozen_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
print(f"\nUnfrozen backbone params: {unfrozen_params:,}")
print("Last 4 blocks of DINOv2 are now trainable ✓")

# Dual learning rate optimizer
optimizer_p2 = optim.AdamW([
    {'params': classifier.parameters(),              'lr': CONFIG['phase2_head_lr']},
    {'params': [p for p in backbone.parameters() if p.requires_grad],
                                                     'lr': CONFIG['phase2_backbone_lr']},
], weight_decay=CONFIG['weight_decay'])

scheduler_p2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer_p2, T_0=8, T_mult=2, eta_min=1e-7
)

# Rebuild data loaders with smaller batch size
train_loader_p2 = DataLoader(
    train_dataset,
    batch_size  = CONFIG['phase2_batch_size'],
    shuffle     = True,
    num_workers = CONFIG['num_workers'],
    pin_memory  = True,
    persistent_workers=True
)
val_loader_p2 = DataLoader(
    val_dataset,
    batch_size  = CONFIG['phase2_batch_size'],
    shuffle     = False,
    num_workers = CONFIG['num_workers'],
    pin_memory  = True,
    persistent_workers=True
)

print("\n" + "="*60)
print("PHASE 2: Training with PARTIALLY UNFROZEN backbone")
print(f"Epochs: {CONFIG['phase2_epochs']}  |  Batch: {CONFIG['phase2_batch_size']}")
print(f"Head LR: {CONFIG['phase2_head_lr']}  |  Backbone LR: {CONFIG['phase2_backbone_lr']}")
print("="*60)

history_p2 = {
    'train_loss': [], 'val_loss': [],
    'train_iou':  [], 'val_iou':  [],
    'train_dice': [], 'val_dice': [],
    'lr': []
}

best_val_iou_p2  = 0.0
patience_counter = 0
scaler_p2        = GradScaler()

for epoch in range(CONFIG['phase2_epochs']):
    global_epoch = CONFIG['phase1_epochs'] + epoch

    # ── Training ─────────────────────────────────────────────
    classifier.train()
    backbone.train()   # backbone now trains too (last 4 blocks)
    train_losses = []
    loss_fn = get_phased_loss(global_epoch)

    pbar = tqdm(train_loader_p2, desc=f"P2 Epoch {epoch+1}/{CONFIG['phase2_epochs']} [Train]", leave=False)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer_p2.zero_grad()
        with autocast():
            feats   = backbone.forward_features(imgs)['x_norm_patchtokens']
            logits  = classifier(feats)
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode='bilinear', align_corners=False)
            loss    = loss_fn(outputs, masks)

        scaler_p2.scale(loss).backward()
        scaler_p2.unscale_(optimizer_p2)
        torch.nn.utils.clip_grad_norm_(
            list(classifier.parameters()) + [p for p in backbone.parameters() if p.requires_grad],
            CONFIG['gradient_clip']
        )
        scaler_p2.step(optimizer_p2)
        scaler_p2.update()

        train_losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    scheduler_p2.step()

    # ── Validation ───────────────────────────────────────────
    classifier.eval()
    backbone.eval()
    val_losses, val_ious = [], []

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader_p2, desc=f"P2 Epoch {epoch+1} [Val]", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            feats   = backbone.forward_features(imgs)['x_norm_patchtokens']
            logits  = classifier(feats)
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode='bilinear', align_corners=False)
            loss    = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)(outputs, masks)
            iou, iou_list = compute_iou(outputs, masks)
            val_losses.append(loss.item())
            val_ious.append(iou)

    epoch_val_iou = np.nanmean(val_ious)

    history_p2['train_loss'].append(np.mean(train_losses))
    history_p2['val_loss'].append(np.mean(val_losses))
    history_p2['train_iou'].append(0.0)
    history_p2['val_iou'].append(float(epoch_val_iou))
    history_p2['train_dice'].append(0.0)
    history_p2['val_dice'].append(0.0)
    history_p2['lr'].append(optimizer_p2.param_groups[0]['lr'])

    print(f"\nEpoch {epoch+1:02d}/{CONFIG['phase2_epochs']} | "
          f"Train Loss: {np.mean(train_losses):.4f} | "
          f"Val Loss: {np.mean(val_losses):.4f} | "
          f"Val mIoU: {epoch_val_iou:.4f}")

    if epoch_val_iou > best_val_iou_p2:
        best_val_iou_p2 = epoch_val_iou
        patience_counter = 0
        torch.save(classifier.state_dict(),
                   f"{CONFIG['output_dir']}/model_finetuned_best.pth")
        print(f"  ★ New best Val mIoU: {best_val_iou_p2:.4f}  → saved model_finetuned_best.pth")
        compute_per_class_iou_report(iou_list)
    else:
        patience_counter += 1
        if patience_counter >= CONFIG['early_stop_patience']:
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            break

    if (epoch + 1) % CONFIG['checkpoint_every'] == 0:
        save_checkpoint({
            'epoch': global_epoch,
            'classifier_state': classifier.state_dict(),
        }, f"{CONFIG['output_dir']}/checkpoints/phase2_epoch{epoch+1}.pth")

    if (epoch + 1) % 5 == 0:
        torch.cuda.empty_cache()

save_training_plots(history_p2, CONFIG['output_dir'], tag='_phase2')

print("\n" + "="*60)
print("✅ PHASE 2 COMPLETE")
print(f"   Phase 1 best mIoU : {best_val_iou_p1:.4f}")
print(f"   Phase 2 best mIoU : {best_val_iou_p2:.4f}")
print(f"   Improvement       : +{best_val_iou_p2 - best_val_iou_p1:.4f}")
print(f"   Weights saved to  : {CONFIG['output_dir']}/model_finetuned_best.pth")
print("="*60)

# ============================================================
# CELL 15 — REPORT MEMBER 1 RESULT TO TEAM
# ============================================================
print("\n📊 MEMBER 1 RESULT — REPORT TO TEAM:")
print(f"   Strategy          : DINOv2 Fine-Tuning (last 4 blocks)")
print(f"   Final Val mIoU    : {best_val_iou_p2:.4f}")
print(f"   Model file        : model_finetuned_best.pth")
print(f"   Share this number with Member 2 and Member 3 for ablation table")
