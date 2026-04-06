"""
╔══════════════════════════════════════════════════════════════════╗
║  TASK 2 — augmentations.py                                       ║
║  Standalone augmentation module for all team members              ║
║  Member 2 — Duality AI Hackathon                                  ║
║                                                                    ║
║  Usage:                                                           ║
║    from augmentations import get_train_transform, get_val_transform║
╚══════════════════════════════════════════════════════════════════╝
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# ================================================================
# CONSTANTS
# ================================================================
IMG_H = 252     # (540 / 2) rounded to nearest ×14
IMG_W = 462     # (960 / 2) rounded to nearest ×14

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ================================================================
# TRAIN TRANSFORM — 12 targeted augmentations
# ================================================================

def get_train_transform(img_h=IMG_H, img_w=IMG_W):
    """
    Returns Albumentations Compose for training.
    
    Design choices:
    - ❌ NO vertical flip: sky must stay at top of image
    - ❌ NO elastic distortion: distorts object boundaries unrealistically
    - ❌ NO MixUp: doesn't work with per-pixel labels
    - ✅ RandomFog + RandomShadow: simulates desert haze / different lighting
    - ✅ CoarseDropout: forces model to use context, not just local features
    - ✅ RandomResizedCrop: simulates camera zoom + viewpoint diversity
    
    All transforms are applied to BOTH image and mask simultaneously
    via albumentations' built-in mask handling.
    """
    return A.Compose([
        # ── Tier 1: GEOMETRIC (simulates camera viewpoint changes) ──
        A.HorizontalFlip(p=0.5),
        # NO vertical flip — sky must stay on top
        
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.3,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7
        ),
        
        A.RandomResizedCrop(
            height=img_h, width=img_w,
            scale=(0.5, 1.5),
            ratio=(0.75, 1.33),
            interpolation=cv2.INTER_LINEAR,
            p=1.0
        ),
        
        A.Perspective(
            scale=(0.02, 0.05),
            keep_size=True,
            p=0.2
        ),
        
        # ── Tier 2: PHOTOMETRIC (simulates different desert conditions) ──
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
            shadow_roi=(0, 0.5, 1, 1),   # shadows only on bottom half (ground)
            num_shadows_lower=1,
            num_shadows_upper=2,
            shadow_dimension=5,
            p=0.2
        ),
        
        A.RandomFog(
            fog_coef_lower=0.1,
            fog_coef_upper=0.25,
            alpha_coef=0.1,
            p=0.1
        ),
        
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3
        ),
        
        # ── Tier 3: OCCLUSION (forces model to use context) ──
        A.CoarseDropout(
            max_holes=6,
            max_height=40,
            max_width=40,
            min_holes=1,
            min_height=10,
            min_width=10,
            fill_value=0,
            mask_fill_value=0,    # fill dropped mask regions with background (0)
            p=0.15
        ),
        
        A.ToGray(p=0.05),    # occasional grayscale — forces model beyond color
        
        # ── ALWAYS: final resize + normalize ──
        A.Resize(height=img_h, width=img_w, p=1.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ================================================================
# VALIDATION / TEST TRANSFORM — minimal, deterministic
# ================================================================

def get_val_transform(img_h=IMG_H, img_w=IMG_W):
    """
    Returns Albumentations Compose for validation and testing.
    No augmentation — just resize, normalize, and convert to tensor.
    """
    return A.Compose([
        A.Resize(height=img_h, width=img_w),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ================================================================
# CUTMIX (batch-level augmentation)
# ================================================================

import numpy as np
import torch
import random


def cutmix_batch(images, masks, alpha=1.0, p=0.25):
    """
    CutMix: cut a random rectangular region from one image and paste
    it onto another in the same batch. Masks are cut-and-pasted too.
    
    Args:
        images: [B, C, H, W] tensor
        masks:  [B, H, W] tensor
        alpha:  Beta distribution parameter (1.0 = uniform)
        p:      Probability of applying CutMix
    
    Returns:
        Modified images and masks.
    """
    if random.random() > p:
        return images, masks
    
    B, C, H, W = images.shape
    
    # Sample lambda from Beta distribution
    lam = np.random.beta(alpha, alpha)
    
    # Random permutation for mixing
    indices = torch.randperm(B)
    
    # Get random box
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    
    cy = random.randint(0, H)
    cx = random.randint(0, W)
    
    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)
    
    # Apply CutMix
    images_out = images.clone()
    masks_out = masks.clone()
    
    images_out[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
    masks_out[:, y1:y2, x1:x2] = masks[indices, y1:y2, x1:x2]
    
    return images_out, masks_out


# ================================================================
# SELF-TEST
# ================================================================

if __name__ == '__main__':
    print("Testing augmentation pipeline...")
    
    train_t = get_train_transform()
    val_t = get_val_transform()
    
    # Create dummy image and mask
    dummy_img = np.random.randint(0, 255, (540, 960, 3), dtype=np.uint8)
    dummy_mask = np.random.randint(0, 10, (540, 960), dtype=np.uint8)
    
    # Apply train transform
    result = train_t(image=dummy_img, mask=dummy_mask)
    img_out = result['image']
    mask_out = result['mask']
    
    print(f"  Train: image={img_out.shape}, mask={mask_out.shape}")
    print(f"  Train mask unique: {np.unique(mask_out)}")
    assert img_out.shape == (3, IMG_H, IMG_W), f"Bad image shape: {img_out.shape}"
    assert mask_out.shape == (IMG_H, IMG_W), f"Bad mask shape: {mask_out.shape}"
    
    # Apply val transform
    result_v = val_t(image=dummy_img, mask=dummy_mask)
    img_v = result_v['image']
    mask_v = result_v['mask']
    
    print(f"  Val:   image={img_v.shape}, mask={mask_v.shape}")
    
    # Test CutMix
    batch_imgs = torch.randn(4, 3, IMG_H, IMG_W)
    batch_masks = torch.randint(0, 10, (4, IMG_H, IMG_W))
    cm_imgs, cm_masks = cutmix_batch(batch_imgs, batch_masks, p=1.0)
    print(f"  CutMix: imgs={cm_imgs.shape}, masks={cm_masks.shape}")
    
    print("\n✅ All augmentation tests passed!")
