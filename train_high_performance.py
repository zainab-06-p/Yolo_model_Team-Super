"""
High-Performance Desert Segmentation Model
Target: mIoU >= 0.90
Uses DINOv2-large + Advanced Decoder + Extensive Augmentation
"""

import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION - HIGH PERFORMANCE
# ═══════════════════════════════════════════════════════════════

class Config:
    # Model Architecture
    BACKBONE = 'dinov2_vitl14'  # Options: vits14, vitb14, vitl14, vitg14
    EMBED_DIM = 1024  # 384 for vits14, 768 for vitb14, 1024 for vitl14
    
    # Image Size - DINOv2 uses patch size 14
    IMG_H = 294  # 21 * 14
    IMG_W = 462  # 33 * 14
    TOKEN_H = 21
    TOKEN_W = 33
    
    # Training
    BATCH_SIZE = 2  # Smaller for large model
    NUM_EPOCHS = 100
    LR = 1e-4
    WEIGHT_DECAY = 0.05
    
    # Classes
    NUM_CLASSES = 10
    
    # Augmentation - EXTENSIVE
    USE_AUGMENTATION = True
    AUG_PROB = 0.8
    
    # Deep Supervision
    USE_DEEP_SUPERVISION = True
    
    # Data
    TRAIN_DIR = 'desert-kaggle-api/train'
    VAL_DIR = 'desert-kaggle-api/val'
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Target: mIoU >= 0.90")
print(f"Backbone: {Config.BACKBONE} (embed_dim={Config.EMBED_DIM})")
print(f"Device: {Config.DEVICE}")

# ═══════════════════════════════════════════════════════════════
# ADVANCED MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════

class AttentionBlock(nn.Module):
    """Self-attention for better feature refinement."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.size()
        
        # Query, Key, Value
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)
        
        # Attention
        attention = torch.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        return self.gamma * out + x

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling for multi-scale context."""
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        
        # Different dilation rates
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv_6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv_12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv_18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Global pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        
        x1 = self.conv_1(x)
        x6 = self.conv_6(x)
        x12 = self.conv_12(x)
        x18 = self.conv_18(x)
        x_global = self.global_pool(x)
        x_global = F.interpolate(x_global, size=size, mode='bilinear', align_corners=False)
        
        x_concat = torch.cat([x1, x6, x12, x18, x_global], dim=1)
        return self.project(x_concat)

class HighPerformanceDecoder(nn.Module):
    """
    Advanced decoder with:
    - ASPP for multi-scale context
    - Attention blocks for feature refinement
    - Deep supervision
    """
    def __init__(self, embed_dim, num_classes, token_h, token_w):
        super().__init__()
        self.token_h = token_h
        self.token_w = token_w
        
        # Project from embedding to feature space
        self.stem = nn.Sequential(
            nn.Conv2d(embed_dim, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # ASPP module
        self.aspp = ASPP(512, 256)
        
        # Attention refinement
        self.attention1 = AttentionBlock(256)
        self.attention2 = AttentionBlock(256)
        
        # Decoder stages
        self.decoder1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Final classifier
        self.classifier = nn.Conv2d(128, num_classes, 1)
        
        # Deep supervision heads
        if Config.USE_DEEP_SUPERVISION:
            self.aux_classifier1 = nn.Conv2d(256, num_classes, 1)
            self.aux_classifier2 = nn.Conv2d(128, num_classes, 1)
    
    def forward(self, x):
        # x: [B, N, C] where N = token_h * token_w
        B, N, C = x.shape
        
        # Reshape to spatial
        x = x.reshape(B, self.token_h, self.token_w, C).permute(0, 3, 1, 2)
        
        # Stem projection
        x = self.stem(x)
        
        # ASPP
        x = self.aspp(x)
        
        # Attention refinement
        x = self.attention1(x)
        x = self.decoder1(x)
        x = self.attention2(x)
        
        aux1 = None
        if Config.USE_DEEP_SUPERVISION and self.training:
            aux1 = self.aux_classifier1(x)
            aux1 = F.interpolate(aux1, size=(Config.IMG_H, Config.IMG_W), 
                                mode='bilinear', align_corners=False)
        
        # Decoder stage 2
        x = self.decoder2(x)
        
        aux2 = None
        if Config.USE_DEEP_SUPERVISION and self.training:
            aux2 = self.aux_classifier2(x)
            aux2 = F.interpolate(aux2, size=(Config.IMG_H, Config.IMG_W),
                                mode='bilinear', align_corners=False)
        
        # Final classification
        x = self.classifier(x)
        
        # Upsample to full resolution
        x = F.interpolate(x, size=(Config.IMG_H, Config.IMG_W),
                         mode='bilinear', align_corners=False)
        
        if Config.USE_DEEP_SUPERVISION and self.training:
            return x, aux1, aux2
        return x

# ═══════════════════════════════════════════════════════════════
# LOAD BACKBONE
# ═══════════════════════════════════════════════════════════════

print("\nLoading DINOv2 backbone...")
backbone = torch.hub.load('facebookresearch/dinov2', Config.BACKBONE,
                          pretrained=True, verbose=False)
backbone = backbone.to(Config.DEVICE).eval()
for p in backbone.parameters():
    p.requires_grad = False
print(f"✓ {Config.BACKBONE} loaded")

# Probe embedding dimension
with torch.no_grad():
    probe = torch.zeros(1, 3, Config.IMG_H, Config.IMG_W, device=Config.DEVICE)
    feat = backbone.forward_features(probe)['x_norm_patchtokens']
    actual_embed_dim = feat.shape[2]
    print(f"  Actual embed dim: {actual_embed_dim}")
    Config.EMBED_DIM = actual_embed_dim

# Create decoder
print("\nCreating high-performance decoder...")
model = HighPerformanceDecoder(Config.EMBED_DIM, Config.NUM_CLASSES, 
                               Config.TOKEN_H, Config.TOKEN_W).to(Config.DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params: {total_params:,}")
print(f"  Trainable: {trainable_params:,}")

# ═══════════════════════════════════════════════════════════════
# EXTENSIVE DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════

class StrongAugmentation:
    """Extensive augmentation for robustness."""
    
    def __init__(self):
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )
    
    def __call__(self, img, mask):
        if random.random() > Config.AUG_PROB:
            return img, mask
        
        img_np = np.array(img)
        mask_np = np.array(mask)
        
        # Random horizontal flip
        if random.random() > 0.5:
            img_np = np.fliplr(img_np)
            mask_np = np.fliplr(mask_np)
        
        # Random rotation
        if random.random() > 0.7:
            angle = random.randint(-15, 15)
            h, w = img_np.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_np = cv2.warpAffine(img_np, M, (w, h), 
                                   borderMode=cv2.BORDER_REFLECT_101)
            mask_np = cv2.warpAffine(mask_np, M, (w, h),
                                    borderMode=cv2.BORDER_REFLECT_101,
                                    flags=cv2.INTER_NEAREST)
        
        # Random scale
        if random.random() > 0.7:
            scale = random.uniform(0.8, 1.2)
            h, w = img_np.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            img_np = cv2.resize(img_np, (new_w, new_h))
            mask_np = cv2.resize(mask_np, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # Crop or pad to original size
            if scale > 1.0:
                # Crop center
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                img_np = img_np[start_h:start_h+h, start_w:start_w+w]
                mask_np = mask_np[start_h:start_h+h, start_w:start_w+w]
            else:
                # Pad
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                img_np = cv2.copyMakeBorder(img_np, pad_h, h-new_h-pad_h, 
                                            pad_w, w-new_w-pad_w,
                                            cv2.BORDER_REFLECT_101)
                mask_np = cv2.copyMakeBorder(mask_np, pad_h, h-new_h-pad_h,
                                            pad_w, w-new_w-pad_w,
                                            cv2.BORDER_REFLECT_101)
        
        # Add noise
        if random.random() > 0.8:
            noise_type = random.choice(['gaussian', 'salt_pepper'])
            if noise_type == 'gaussian':
                noise = np.random.normal(0, 15, img_np.shape).astype(np.int16)
                img_np = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            else:
                # Salt and pepper
                amount = 0.01
                num_pixels = int(amount * img_np.size)
                coords = [np.random.randint(0, i, num_pixels) for i in img_np.shape[:2]]
                img_np[coords[0], coords[1]] = 255
                coords = [np.random.randint(0, i, num_pixels) for i in img_np.shape[:2]]
                img_np[coords[0], coords[1]] = 0
        
        # Gaussian blur
        if random.random() > 0.8:
            kernel_size = random.choice([3, 5])
            img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
        
        # Color jitter (convert to PIL and back)
        img = Image.fromarray(img_np)
        img = self.color_jitter(img)
        img_np = np.array(img)
        
        return Image.fromarray(img_np), Image.fromarray(mask_np)

# Transforms
img_transform = transforms.Compose([
    transforms.Resize((Config.IMG_H, Config.IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((Config.IMG_H, Config.IMG_W), 
                      interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor()
])

print("\n✓ Strong augmentation pipeline ready")

# ═══════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════

VALUE_MAP = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

def convert_mask(mask_pil):
    arr = np.array(mask_pil)
    out = np.zeros_like(arr, dtype=np.uint8)
    for raw, cls in VALUE_MAP.items():
        out[arr == raw] = cls
    return out

class DesertDataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.mask_dir = os.path.join(data_dir, 'Segmentation')
        self.filenames = sorted([f for f in os.listdir(self.image_dir) 
                                if f.endswith('.png')])
        self.is_train = is_train
        self.aug = StrongAugmentation() if is_train else None
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        
        img = Image.open(os.path.join(self.image_dir, fname)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, fname))
        
        # Augmentation
        if self.is_train and self.aug:
            img, mask = self.aug(img, mask)
        
        # Convert mask
        mask_cls = Image.fromarray(convert_mask(mask))
        
        # Transform
        img_t = img_transform(img)
        mask_t = (mask_transform(mask_cls) * 255).long().squeeze(0)
        
        return img_t, mask_t

# Create datasets
train_dataset = DesertDataset(Config.TRAIN_DIR, is_train=True)
val_dataset = DesertDataset(Config.VAL_DIR, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                          shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE,
                        shuffle=False, num_workers=2, pin_memory=True)

print(f"Train: {len(train_dataset)} images")
print(f"Val: {len(val_dataset)} images")

# ═══════════════════════════════════════════════════════════════
# LOSS FUNCTION - Advanced
# ═══════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        
        # One-hot encode target
        target_one_hot = F.one_hot(target, num_classes=Config.NUM_CLASSES)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
    
    def forward(self, pred, target):
        # Main loss
        ce = self.ce(pred, target)
        dice = self.dice(pred, target)
        focal = self.focal(pred, target)
        
        # Weighted combination
        return 0.5 * ce + 0.3 * dice + 0.2 * focal

criterion = CombinedLoss().to(Config.DEVICE)
print("\n✓ Combined loss (CE + Dice + Focal) ready")

# ═══════════════════════════════════════════════════════════════
# OPTIMIZER & SCHEDULER
# ═══════════════════════════════════════════════════════════════

optimizer = AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)

scheduler = OneCycleLR(optimizer, max_lr=Config.LR, 
                       steps_per_epoch=len(train_loader),
                       epochs=Config.NUM_EPOCHS,
                       pct_start=0.1,
                       anneal_strategy='cos')

print(f"✓ Optimizer: AdamW (lr={Config.LR})")
print(f"✓ Scheduler: OneCycleLR")

# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def compute_miou(pred, target):
    """Compute mean IoU."""
    pred = pred.argmax(dim=1)
    
    ious = []
    for cls in range(Config.NUM_CLASSES):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())
    
    return np.nanmean(ious)

print("\n" + "="*60)
print("STARTING TRAINING - Target mIoU >= 0.90")
print("="*60)

best_miou = 0.0

for epoch in range(Config.NUM_EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS}"):
        images = images.to(Config.DEVICE)
        masks = masks.to(Config.DEVICE)
        
        optimizer.zero_grad()
        
        # Forward
        with torch.no_grad():
            features = backbone.forward_features(images)['x_norm_patchtokens']
        
        outputs = model(features)
        
        # Compute loss
        if Config.USE_DEEP_SUPERVISION and isinstance(outputs, tuple):
            main_out, aux1, aux2 = outputs
            loss_main = criterion(main_out, masks)
            loss_aux1 = criterion(aux1, masks)
            loss_aux2 = criterion(aux2, masks)
            loss = loss_main + 0.4 * loss_aux1 + 0.4 * loss_aux2
        else:
            loss = criterion(outputs, masks)
        
        # Backward
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_miou = 0.0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(Config.DEVICE)
            masks = masks.to(Config.DEVICE)
            
            features = backbone.forward_features(images)['x_norm_patchtokens']
            outputs = model(features)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            miou = compute_miou(outputs, masks)
            val_miou += miou
    
    avg_val_miou = val_miou / len(val_loader)
    
    print(f"Epoch {epoch+1}: Loss={avg_train_loss:.4f}, Val mIoU={avg_val_miou:.4f}")
    
    # Save best
    if avg_val_miou > best_miou:
        best_miou = avg_val_miou
        torch.save(model.state_dict(), 'best_high_performance_model.pth')
        print(f"  ✓ New best model saved! mIoU={best_miou:.4f}")
    
    # Early stopping check
    if best_miou >= 0.90:
        print(f"\n{'='*60}")
        print(f"🎉 TARGET ACHIEVED! mIoU = {best_miou:.4f}")
        print(f"{'='*60}")
        break

print(f"\n{'='*60}")
print(f"TRAINING COMPLETE")
print(f"Best mIoU: {best_miou:.4f}")
print(f"{'='*60}")
