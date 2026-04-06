"""
Synthetic Black Sky Training Augmentation
Creates training data with black sky to improve model robustness
"""

import os
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from tqdm import tqdm
import random

def load_image_and_mask(image_path, mask_path):
    """Load image and its segmentation mask."""
    img = np.array(Image.open(image_path).convert('RGB'))
    mask = np.array(Image.open(mask_path))
    return img, mask

def identify_sky_region(mask_np):
    """
    Identify sky region from mask (class 9 in VALUE_MAP).
    VALUE_MAP: 10000 -> 9 (Sky)
    """
    # Sky is class 9 in our mapping
    # Original mask uses 10000 for sky
    sky_mask = (mask_np == 10000)
    return sky_mask

def synthetic_black_sky(image_np, mask_np, variant_type='full_black'):
    """
    Create synthetic black sky variations.
    
    variant_type:
        - 'full_black': Complete black sky (0,0,0)
        - 'noisy_black': Black with noise
        - 'gradient_black': Gradient from dark to black
        - 'dark_gray': Dark gray instead of pure black
    """
    result = image_np.copy()
    sky_mask = identify_sky_region(mask_np)
    
    if not np.any(sky_mask):
        # No sky in this image, return original
        return result
    
    h, w = image_np.shape[:2]
    
    if variant_type == 'full_black':
        # Pure black sky
        result[sky_mask] = [0, 0, 0]
        
    elif variant_type == 'noisy_black':
        # Black sky with slight noise
        noise = np.random.randint(0, 15, (h, w, 3))
        result[sky_mask] = noise[sky_mask]
        
    elif variant_type == 'gradient_black':
        # Gradient from top (black) to horizon (dark gray)
        y_coords = np.arange(h)
        # Find horizon line (top of non-sky region)
        sky_rows = np.any(sky_mask, axis=1)
        if np.any(sky_rows):
            horizon_y = np.max(np.where(sky_rows)[0])
        else:
            horizon_y = h // 3
        
        for y in range(h):
            if y <= horizon_y and sky_mask[y].any():
                # Create gradient: darker at top
                intensity = int(30 * (1 - y / (horizon_y + 1)))
                result[y, sky_mask[y]] = [intensity, intensity, intensity]
    
    elif variant_type == 'dark_gray':
        # Dark gray sky instead of pure black
        gray_value = np.random.randint(15, 35)
        result[sky_mask] = [gray_value, gray_value, gray_value]
    
    elif variant_type == 'blue_tinted_black':
        # Very dark blue to distinguish from black background
        result[sky_mask] = [5, 5, 15]  # Dark blue
    
    return result

def darken_objects(image_np, mask_np, darken_factor=0.6):
    """
    Darken non-sky objects to simulate low-light conditions.
    """
    result = image_np.copy()
    sky_mask = identify_sky_region(mask_np)
    non_sky_mask = ~sky_mask
    
    # Darken non-sky regions
    for c in range(3):
        result[:, :, c][non_sky_mask] = (
            result[:, :, c][non_sky_mask] * darken_factor
        ).astype(np.uint8)
    
    return result

def create_monochromatic_variant(image_np, mask_np):
    """
    Create a variant where sky is black and objects are dark.
    This simulates the exact test condition described.
    """
    # Step 1: Make sky black
    result = synthetic_black_sky(image_np, mask_np, 'full_black')
    
    # Step 2: Darken objects significantly
    result = darken_objects(result, mask_np, darken_factor=0.4)
    
    # Step 3: Add slight noise to prevent pure black regions
    noise = np.random.randint(0, 8, result.shape)
    result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return result

def generate_training_augmentations(train_dir, output_dir, num_variants=5):
    """
    Generate synthetic training augmentations with black sky.
    
    Args:
        train_dir: Directory with training images and masks
        output_dir: Directory to save augmented images
        num_variants: Number of variants per original image
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/Color_Images", exist_ok=True)
    os.makedirs(f"{output_dir}/Segmentation", exist_ok=True)
    
    image_dir = os.path.join(train_dir, 'Color_Images')
    mask_dir = os.path.join(train_dir, 'Segmentation')
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    
    print(f"Generating augmentations for {len(image_files)} images...")
    
    # Variant types to generate
    variant_types = [
        'full_black',
        'noisy_black', 
        'gradient_black',
        'dark_gray',
        'monochromatic_dark'
    ]
    
    for fname in tqdm(image_files, desc="Augmenting"):
        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, fname)
        
        if not os.path.exists(mask_path):
            continue
        
        img, mask = load_image_and_mask(img_path, mask_path)
        
        # Generate variants
        for i, vtype in enumerate(variant_types[:num_variants]):
            if vtype == 'monochromatic_dark':
                aug_img = create_monochromatic_variant(img, mask)
            else:
                aug_img = synthetic_black_sky(img, mask, vtype)
            
            # Save augmented image
            aug_fname = f"{Path(fname).stem}_aug{i}_{vtype}.png"
            Image.fromarray(aug_img).save(
                os.path.join(output_dir, 'Color_Images', aug_fname)
            )
            
            # Save mask (same as original)
            Image.fromarray(mask).save(
                os.path.join(output_dir, 'Segmentation', aug_fname)
            )
    
    print(f"\n✓ Generated {len(image_files) * num_variants} augmented images")
    print(f"Saved to: {output_dir}")

def test_black_sky_augmentation():
    """Test the augmentation on sample images."""
    test_images = [
        ("desert-kaggle-api/train/Color_Images/cc0000013.png",
         "desert-kaggle-api/train/Segmentation/cc0000013.png"),
        ("desert-kaggle-api/train/Color_Images/cc0000014.png",
         "desert-kaggle-api/train/Segmentation/cc0000014.png"),
    ]
    
    output_dir = "black_sky_test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path, mask_path in test_images:
        if not os.path.exists(img_path):
            continue
        
        img, mask = load_image_and_mask(img_path, mask_path)
        base_name = Path(img_path).stem
        
        print(f"\nProcessing {base_name}...")
        
        # Save original
        Image.fromarray(img).save(f"{output_dir}/{base_name}_original.png")
        
        # Generate and save variants
        variants = [
            ('full_black', synthetic_black_sky(img, mask, 'full_black')),
            ('noisy_black', synthetic_black_sky(img, mask, 'noisy_black')),
            ('gradient_black', synthetic_black_sky(img, mask, 'gradient_black')),
            ('monochromatic', create_monochromatic_variant(img, mask)),
        ]
        
        for vtype, variant_img in variants:
            Image.fromarray(variant_img).save(
                f"{output_dir}/{base_name}_{vtype}.png"
            )
            print(f"  ✓ Created {vtype} variant")
    
    print(f"\n✓ Test complete. Check {output_dir}/ for results")

# Training configuration for Kaggle
TRAINING_CONFIG = """
# Add this to your training script for black sky robustness:

# 1. Load synthetic black sky data
augmented_dir = '/kaggle/input/black-sky-augmented'

# 2. Mix with original training data (e.g., 20% augmented, 80% original)
train_dataset = ConcatDataset([
    original_train_dataset,
    augmented_dataset
])

# 3. Use class weights to handle imbalance
class_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0]
# Emphasize rare classes (Ground Clutter, Logs) and Sky

# 4. Train with focus on edge detection
# Add edge-aware loss or boundary loss
"""

if __name__ == "__main__":
    print("="*60)
    print("Black Sky Training Augmentation Generator")
    print("="*60)
    
    # Test on sample images
    print("\n1. Testing augmentation on sample images...")
    test_black_sky_augmentation()
    
    # Generate full augmentations if directories exist
    train_dir = "desert-kaggle-api/train"
    if os.path.exists(train_dir):
        print(f"\n2. Generating full training augmentations...")
        output_dir = "desert-kaggle-api/train_augmented_black_sky"
        generate_training_augmentations(train_dir, output_dir, num_variants=5)
    else:
        print(f"\n2. Training directory not found: {train_dir}")
        print("   Skipping full augmentation generation")
    
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS")
    print("="*60)
    print("""
To use for Kaggle training:

1. Run this script to generate synthetic black sky images:
   python black_sky_augmentation.py

2. Upload augmented images to Kaggle as a dataset:
   - Name: 'black-sky-augmented'
   - Contains: Color_Images/ and Segmentation/ folders

3. In your training notebook:
   
   # Load original + augmented data
   from torch.utils.data import ConcatDataset
   
   original_dataset = DesertDataset('/kaggle/input/yolo-training-data/train')
   augmented_dataset = DesertDataset('/kaggle/input/black-sky-augmented')
   
   # Mix datasets (e.g., 1:1 ratio)
   train_dataset = ConcatDataset([original_dataset, augmented_dataset])
   
   train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

4. Train the model - it will now learn to:
   - Distinguish black sky from black background using texture
   - Identify objects in very dark conditions
   - Handle edge cases with monochromatic images

5. For inference, use ensemble_kaggle.py which includes:
   - handle_monochromatic_black() preprocessing
   - Texture-based sky/object separation
   - Edge enhancement for dark images
""")
