"""
╔══════════════════════════════════════════════════════════════════╗
║  TASK 4 — rare_class_tools.py                                    ║
║  Copy-paste augmentor for rare classes (Logs, Ground Clutter)     ║
║  Member 2 — Duality AI Hackathon                                  ║
║                                                                    ║
║  Usage:                                                           ║
║    from rare_class_tools import CopyPasteAugmentor                ║
║    cp = CopyPasteAugmentor(image_dir, mask_dir, pool_size=200)   ║
║    augmented_img, augmented_mask = cp.apply(image, mask)          ║
╚══════════════════════════════════════════════════════════════════╝

WHY THIS EXISTS:
  Logs = 0.07% of all pixels in training data (500x less than Sky).
  Without targeted intervention, the model will NEVER learn to detect Logs.
  
  Copy-paste augmentation:
  1. Pre-scans all training masks for rare class instances
  2. Extracts each instance as an (image_crop, mask_crop, binary_mask) tuple
  3. During training, randomly pastes these crops onto other training images
  4. Result: rare classes appear 5-10x more often → model learns them
"""

import numpy as np
import cv2
import os
import random
from PIL import Image
from tqdm import tqdm

# Import from our project modules
try:
    from dataset import VALUE_MAP, CLASS_NAMES, convert_mask_np
except ImportError:
    # Fallback if running standalone
    VALUE_MAP = {
        0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
        550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
    }
    CLASS_NAMES = [
        'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
        'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
    ]
    def convert_mask_np(mask_pil):
        arr = np.array(mask_pil)
        out = np.zeros(arr.shape, dtype=np.uint8)
        for raw_val, class_id in VALUE_MAP.items():
            out[arr == raw_val] = class_id
        return out


class CopyPasteAugmentor:
    """
    Builds a pool of rare-class instance crops from training data,
    then pastes them into random training images during training.
    
    Example:
        cp = CopyPasteAugmentor(
            image_dir='train/Color_Images',
            mask_dir='train/Segmentation',
            target_class_ids=[5, 6],  # Ground Clutter (5), Logs (6)
            pool_size=200
        )
        
        # In your training loop:
        if random.random() < 0.3:
            image, mask = cp.apply(image, mask, n_pastes=2)
    """
    
    def __init__(self, image_dir, mask_dir, target_class_ids=None,
                 pool_size=200, min_component_area=100, max_scan=None):
        """
        Args:
            image_dir: path to Color_Images/ folder
            mask_dir: path to Segmentation/ folder
            target_class_ids: list of CLASS IDs (0-9) to extract
                              Default: [5, 6] = Ground Clutter, Logs
            pool_size: maximum number of crops to collect
            min_component_area: ignore components smaller than this (pixels)
            max_scan: maximum images to scan (None = all)
        """
        self.pool = []
        self.target_class_ids = target_class_ids or [5, 6]  # Ground Clutter, Logs
        
        self._build_pool(image_dir, mask_dir, pool_size, min_component_area, max_scan)
    
    def _build_pool(self, image_dir, mask_dir, pool_size, min_area, max_scan):
        """Scan training data and extract crops of rare class instances."""
        files = sorted(os.listdir(image_dir))
        if max_scan:
            files = files[:max_scan]
        
        target_names = [CLASS_NAMES[cid] for cid in self.target_class_ids]
        print(f"Building copy-paste pool for: {target_names}")
        print(f"  Target pool size: {pool_size} crops")
        print(f"  Min component area: {min_area} pixels")
        print(f"  Scanning {len(files)} images...")
        
        for fname in tqdm(files, desc="Building pool", leave=False):
            if len(self.pool) >= pool_size:
                break
            
            # Read image and mask
            img_path = os.path.join(image_dir, fname)
            mask_path = os.path.join(mask_dir, fname)
            
            if not os.path.exists(mask_path):
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            mask = convert_mask_np(Image.open(mask_path))
            
            # Find instances of each target class
            for class_id in self.target_class_ids:
                binary = (mask == class_id).astype(np.uint8)
                
                if binary.sum() < min_area:
                    continue
                
                # Find connected components (individual instances)
                num_labels, labels = cv2.connectedComponents(binary)
                
                for label_id in range(1, num_labels):
                    if len(self.pool) >= pool_size:
                        break
                    
                    component = (labels == label_id)
                    
                    # Skip tiny components
                    if component.sum() < min_area:
                        continue
                    
                    # Get bounding box
                    rows = np.where(np.any(component, axis=1))[0]
                    cols = np.where(np.any(component, axis=0))[0]
                    
                    if len(rows) == 0 or len(cols) == 0:
                        continue
                    
                    r0, r1 = rows[0], rows[-1] + 1
                    c0, c1 = cols[0], cols[-1] + 1
                    
                    # Extract crops
                    img_crop = img[r0:r1, c0:c1].copy()
                    mask_crop = mask[r0:r1, c0:c1].copy()
                    binary_crop = component[r0:r1, c0:c1]
                    
                    self.pool.append({
                        'img_crop': img_crop,
                        'mask_crop': mask_crop,
                        'binary': binary_crop,
                        'class_id': class_id,
                        'source_file': fname,
                        'area': int(component.sum()),
                    })
        
        # Summary
        if self.pool:
            counts_per_class = {}
            for crop in self.pool:
                cid = crop['class_id']
                counts_per_class[cid] = counts_per_class.get(cid, 0) + 1
            
            print(f"✓ Pool built: {len(self.pool)} total crops")
            for cid, count in sorted(counts_per_class.items()):
                print(f"  [{cid}] {CLASS_NAMES[cid]}: {count} instances")
        else:
            print("⚠ Pool is EMPTY — no instances found!")
            print("  Check target_class_ids and min_component_area")
    
    def apply(self, image, mask, n_pastes=2):
        """
        Paste random rare-class crops onto an image/mask pair.
        
        Args:
            image: numpy array [H, W, 3] uint8
            mask:  numpy array [H, W] uint8
            n_pastes: number of crops to paste (default 2)
        
        Returns:
            modified_image, modified_mask (numpy arrays)
        """
        if not self.pool:
            return image, mask
        
        H, W = image.shape[:2]
        result_img = image.copy()
        result_mask = mask.copy()
        
        for _ in range(n_pastes):
            crop = random.choice(self.pool)
            ih, iw = crop['img_crop'].shape[:2]
            
            # Skip if crop is larger than image
            if ih >= H or iw >= W:
                continue
            
            # Random position
            py = random.randint(0, H - ih)
            px = random.randint(0, W - iw)
            
            # Paste only within the component's binary mask (clean edges)
            b = crop['binary']
            result_img[py:py+ih, px:px+iw][b] = crop['img_crop'][b]
            result_mask[py:py+ih, px:px+iw][b] = crop['mask_crop'][b]
        
        return result_img, result_mask
    
    def apply_with_scale(self, image, mask, n_pastes=2, scale_range=(0.5, 1.5)):
        """
        Same as apply(), but randomly scales each crop before pasting.
        This creates even more variation from a small pool.
        """
        if not self.pool:
            return image, mask
        
        H, W = image.shape[:2]
        result_img = image.copy()
        result_mask = mask.copy()
        
        for _ in range(n_pastes):
            crop = random.choice(self.pool)
            
            # Random scale
            scale = random.uniform(*scale_range)
            ih = int(crop['img_crop'].shape[0] * scale)
            iw = int(crop['img_crop'].shape[1] * scale)
            
            if ih < 10 or iw < 10 or ih >= H or iw >= W:
                continue
            
            # Resize crop, mask, and binary
            img_scaled = cv2.resize(crop['img_crop'], (iw, ih), interpolation=cv2.INTER_LINEAR)
            mask_scaled = cv2.resize(crop['mask_crop'], (iw, ih), interpolation=cv2.INTER_NEAREST)
            binary_scaled = cv2.resize(crop['binary'].astype(np.uint8), (iw, ih),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
            
            # Random position
            py = random.randint(0, H - ih)
            px = random.randint(0, W - iw)
            
            result_img[py:py+ih, px:px+iw][binary_scaled] = img_scaled[binary_scaled]
            result_mask[py:py+ih, px:px+iw][binary_scaled] = mask_scaled[binary_scaled]
        
        return result_img, result_mask
    
    def visualize(self, n=5, save_path=None):
        """Visualize random crops from the pool."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        if not self.pool:
            print("Pool is empty, nothing to visualize")
            return
        
        samples = random.sample(self.pool, min(n, len(self.pool)))
        
        fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n))
        if n == 1:
            axes = axes.reshape(1, -1)
        
        for i, crop in enumerate(samples):
            axes[i, 0].imshow(crop['img_crop'])
            axes[i, 0].set_title(f"Image crop | {CLASS_NAMES[crop['class_id']]}")
            
            axes[i, 1].imshow(crop['mask_crop'], vmin=0, vmax=9, cmap='tab10')
            axes[i, 1].set_title(f"Mask crop | area={crop['area']}")
            
            axes[i, 2].imshow(crop['binary'], cmap='gray')
            axes[i, 2].set_title(f"Binary mask | from {crop['source_file']}")
        
        for ax in axes.flat:
            ax.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=100)
            print(f"✓ Saved visualization: {save_path}")
        plt.close()


# ================================================================
# SELF-TEST
# ================================================================

if __name__ == '__main__':
    print("Testing rare_class_tools...")
    
    # Test with dummy data
    test_img = np.random.randint(0, 255, (540, 960, 3), dtype=np.uint8)
    test_mask = np.ones((540, 960), dtype=np.uint8) * 8  # all Landscape
    
    # Place a fake "Log" region
    test_mask[100:150, 200:280] = 6  # Logs
    
    # Create a minimal copy-paste augmentor with manual pool
    cp = CopyPasteAugmentor.__new__(CopyPasteAugmentor)
    cp.pool = [{
        'img_crop': np.random.randint(0, 255, (50, 80, 3), dtype=np.uint8),
        'mask_crop': np.full((50, 80), 6, dtype=np.uint8),  # all Logs
        'binary': np.ones((50, 80), dtype=bool),
        'class_id': 6,
        'source_file': 'test.png',
        'area': 4000,
    }]
    cp.target_class_ids = [5, 6]
    
    # Apply
    aug_img, aug_mask = cp.apply(test_img, test_mask, n_pastes=3)
    
    # Verify
    original_logs = (test_mask == 6).sum()
    augmented_logs = (aug_mask == 6).sum()
    
    print(f"  Original Logs pixels: {original_logs}")
    print(f"  After copy-paste:     {augmented_logs}")
    assert augmented_logs >= original_logs, "Copy-paste should add more rare class pixels"
    print("✅ Rare class tools test passed!")
