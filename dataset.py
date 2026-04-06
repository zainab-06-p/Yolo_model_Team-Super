"""
╔══════════════════════════════════════════════════════════════════╗
║  TASK 3 — dataset.py                                             ║
║  Unified dataset class for all team members                       ║
║  Member 2 — Duality AI Hackathon                                  ║
║                                                                    ║
║  Usage:                                                           ║
║    from dataset import build_dataloaders, AugmentedDesertDataset  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import cv2
from tqdm import tqdm

# ================================================================
# CLASS MAPPING — matches the provided train_segmentation.py
# ================================================================
VALUE_MAP = {
    0: 0,        # Background (phantom — not in actual data)
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    700: 6,      # Logs        ← 0.07% of pixels — rarest class!
    800: 7,      # Rocks
    7100: 8,     # Landscape
    10000: 9     # Sky
}

CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

N_CLASSES = 10

# Classes that appear in < 1% of pixels — need oversampling
RARE_CLASSES = [5, 6]  # Ground Clutter (4.70%), Logs (0.07%)
# Note: Dry Bushes (1.04%) and Flowers (2.57% but unmapped) are also uncommon


# ================================================================
# MASK CONVERSION
# ================================================================

def convert_mask_np(mask_pil):
    """
    Convert 16-bit PIL mask (raw pixel values like 100, 200, 7100)
    to uint8 class IDs (0–9).
    
    CRITICAL: This replaces the original script's convert_mask() which
    returned a PIL Image. We return numpy array directly.
    
    Args:
        mask_pil: PIL Image in mode 'I;16' (16-bit)
    Returns:
        numpy array of shape (H, W), dtype=uint8, values 0–9
    """
    arr = np.array(mask_pil)
    out = np.zeros(arr.shape, dtype=np.uint8)
    for raw_val, class_id in VALUE_MAP.items():
        out[arr == raw_val] = class_id
    return out


# ================================================================
# DATASET CLASS
# ================================================================

class AugmentedDesertDataset(Dataset):
    """
    Desert segmentation dataset with Albumentations support.
    
    Key improvements over original MaskDataset:
    1. Uses Albumentations → image and mask get SAME spatial transforms
    2. Reads masks as 16-bit correctly via PIL
    3. Converts raw mask values to class IDs using VALUE_MAP
    4. Optionally builds sampler weights for rare class oversampling
    5. Returns mask as LongTensor (not multiplied by 255!)
    
    Fix for original script bug:
    Original: mask = self.mask_transform(mask) * 255
    This is WRONG because after ToTensor (which divides by 255),
    multiplying by 255 is an undo hack. Our version is clean.
    """
    
    def __init__(self, data_dir, transform=None, build_sampler_weights=False,
                 copy_paste_augmentor=None, copy_paste_prob=0.3):
        """
        Args:
            data_dir: path containing 'Color_Images/' and 'Segmentation/' folders
            transform: albumentations.Compose or None
            build_sampler_weights: if True, scan masks for rare classes and
                                   assign 3x weight to images containing them
            copy_paste_augmentor: CopyPasteAugmentor instance or None
            copy_paste_prob: probability of applying copy-paste per sample
        """
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.mask_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.copy_paste = copy_paste_augmentor
        self.copy_paste_prob = copy_paste_prob
        
        # List and sort filenames
        self.filenames = sorted(os.listdir(self.image_dir))
        
        # Filter to only files that exist in BOTH Color_Images and Segmentation
        mask_files = set(os.listdir(self.mask_dir))
        self.filenames = [f for f in self.filenames if f in mask_files]
        
        # Build sampler weights if requested
        self.sampler_weights = None
        if build_sampler_weights:
            self._build_sampler_weights()
    
    def _build_sampler_weights(self):
        """
        Scan all masks and give 3x sampling weight to images containing rare classes.
        This means during training, images with Logs/Ground Clutter appear 3x more often.
        """
        print("Building sampler weights (scanning for rare classes)...")
        weights = []
        rare_raw_values = [raw for raw, cid in VALUE_MAP.items() if cid in RARE_CLASSES]
        
        for fname in tqdm(self.filenames, desc="Scanning masks", leave=False):
            mask = np.array(Image.open(os.path.join(self.mask_dir, fname)))
            has_rare = any((mask == raw_val).any() for raw_val in rare_raw_values)
            weights.append(3.0 if has_rare else 1.0)
        
        self.sampler_weights = weights
        rare_count = sum(1 for w in weights if w > 1.0)
        print(f"✓ Images with rare classes: {rare_count}/{len(self.filenames)} "
              f"({rare_count/len(self.filenames)*100:.1f}%)")
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        
        # Read image as RGB numpy array
        img_path = os.path.join(self.image_dir, fname)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read mask as 16-bit and convert to class IDs
        mask_path = os.path.join(self.mask_dir, fname)
        mask_pil = Image.open(mask_path)
        mask = convert_mask_np(mask_pil)
        
        # Apply copy-paste BEFORE other augmentations (on raw numpy)
        if self.copy_paste is not None:
            import random
            if random.random() < self.copy_paste_prob:
                image, mask = self.copy_paste.apply(image, mask, n_pastes=2)
        
        # Apply albumentations transforms
        if self.transform:
            result = self.transform(image=image, mask=mask)
            image = result['image']          # [3, H, W] float tensor, normalized
            mask = torch.tensor(result['mask'], dtype=torch.long)  # [H, W] long tensor
        else:
            # Fallback: just resize and convert
            image = cv2.resize(image, (462, 252))
            mask = cv2.resize(mask, (462, 252), interpolation=cv2.INTER_NEAREST)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
        
        return image, mask


class TestDesertDataset(Dataset):
    """
    Test set dataset — returns image, mask, AND filename.
    Filename is needed to save predictions with matching names.
    """
    
    def __init__(self, data_dir, transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.mask_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.filenames = sorted(os.listdir(self.image_dir))
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        
        # Read image
        image = cv2.imread(os.path.join(self.image_dir, fname))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read mask (may or may not be ground truth in test set)
        mask_path = os.path.join(self.mask_dir, fname)
        if os.path.exists(mask_path):
            mask = convert_mask_np(Image.open(mask_path))
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        if self.transform:
            result = self.transform(image=image, mask=mask)
            image = result['image']
            mask = torch.tensor(result['mask'], dtype=torch.long)
        
        return image, mask, fname


# ================================================================
# DATALOADER BUILDER
# ================================================================

def build_dataloaders(train_dir, val_dir, train_transform, val_transform,
                      batch_size=8, num_workers=2, use_weighted_sampler=True,
                      copy_paste_augmentor=None):
    """
    Build train and validation DataLoaders with all bells and whistles.
    
    Args:
        train_dir: path to train split (contains Color_Images/ and Segmentation/)
        val_dir: path to val split
        train_transform: albumentations Compose for training
        val_transform: albumentations Compose for validation
        batch_size: batch size (8 for frozen backbone, 4 for unfrozen)
        num_workers: data loading workers
        use_weighted_sampler: oversample rare class images
        copy_paste_augmentor: CopyPasteAugmentor or None
    
    Returns:
        train_loader, val_loader, train_dataset
    """
    # Build datasets
    train_dataset = AugmentedDesertDataset(
        train_dir,
        transform=train_transform,
        build_sampler_weights=use_weighted_sampler,
        copy_paste_augmentor=copy_paste_augmentor,
        copy_paste_prob=0.3 if copy_paste_augmentor else 0.0
    )
    
    val_dataset = AugmentedDesertDataset(
        val_dir,
        transform=val_transform,
        build_sampler_weights=False
    )
    
    # Build samplers
    if use_weighted_sampler and train_dataset.sampler_weights:
        sampler = WeightedRandomSampler(
            weights=train_dataset.sampler_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        shuffle = False  # can't use both sampler and shuffle
    else:
        sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True  # avoid batch size 1 which can break BatchNorm
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    
    print(f"✓ DataLoaders built:")
    print(f"  Train: {len(train_dataset)} images, {len(train_loader)} batches (batch_size={batch_size})")
    print(f"  Val:   {len(val_dataset)} images, {len(val_loader)} batches")
    
    return train_loader, val_loader, train_dataset


# ================================================================
# SELF-TEST
# ================================================================

if __name__ == '__main__':
    print("Testing dataset module...")
    
    # Test mask conversion
    test_mask = np.array([
        [100, 200, 300],
        [7100, 10000, 550],
        [700, 800, 500]
    ], dtype=np.uint16)
    
    from PIL import Image as PILImage
    pil_mask = PILImage.fromarray(test_mask)
    converted = convert_mask_np(pil_mask)
    
    expected = np.array([
        [1, 2, 3],
        [8, 9, 5],
        [6, 7, 4]
    ], dtype=np.uint8)
    
    assert np.array_equal(converted, expected), f"Mask conversion failed!\nGot:\n{converted}\nExpected:\n{expected}"
    print("✅ Mask conversion test passed!")
    
    print(f"\nClass mapping:")
    for raw_val, class_id in sorted(VALUE_MAP.items()):
        print(f"  {raw_val:>6} → {class_id} ({CLASS_NAMES[class_id]})")
