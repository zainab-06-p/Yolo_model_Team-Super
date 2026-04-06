"""
Monochromatic Black Image Handler
Specialized preprocessing for images with black sky and dark objects
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_black_content(image_np):
    """
    Analyze how much of the image is black/monochromatic.
    Returns percentage of black pixels and dark regions.
    """
    if len(image_np.shape) == 3:
        # Check for near-black pixels (all channels < 20)
        black_mask = np.all(image_np < 20, axis=2)
        dark_mask = np.all(image_np < 50, axis=2)
        
        black_percent = np.sum(black_mask) / black_mask.size * 100
        dark_percent = np.sum(dark_mask) / dark_mask.size * 100
        
        # Check if it's monochromatic (all channels similar)
        channel_diff = np.std(image_np, axis=2)
        mono_mask = channel_diff < 10
        mono_percent = np.sum(mono_mask) / mono_mask.size * 100
        
        return {
            'black_percent': black_percent,
            'dark_percent': dark_percent,
            'mono_percent': mono_percent,
            'is_monochromatic': mono_percent > 30,
            'is_dark': dark_percent > 40
        }
    return {'black_percent': 0, 'dark_percent': 0, 'mono_percent': 0, 
            'is_monochromatic': False, 'is_dark': False}

def enhance_monochromatic_image(image_np):
    """
    Specialized enhancement for monochromatic black images.
    Uses edge detection and local contrast to separate objects.
    """
    # Convert to grayscale for analysis
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
    
    # Step 1: Local histogram equalization for dark regions
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    
    # Step 2: Edge detection to find object boundaries
    edges = cv2.Canny(gray_enhanced, 50, 150)
    
    # Step 3: Morphological operations to enhance edges
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Step 4: Create edge mask for enhancement
    edge_mask = edges_dilated > 0
    
    # Step 5: Adaptive thresholding for different regions
    thresh = cv2.adaptiveThreshold(gray_enhanced, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Step 6: Enhance color channels based on luminance
    if len(image_np.shape) == 3:
        # Convert to LAB
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Enhance L channel with local contrast
        l_enhanced = clahe.apply(l)
        
        # Boost color channels (a, b) to differentiate classes
        a_boosted = cv2.addWeighted(a, 1.5, a, 0, 0)
        b_boosted = cv2.addWeighted(b, 1.5, b, 0, 0)
        
        # Merge back
        lab_enhanced = cv2.merge([l_enhanced, a_boosted, b_boosted])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        # Step 7: Enhance edges in color image
        result[edge_mask] = cv2.addWeighted(result[edge_mask], 1.2, 
                                            result[edge_mask], 0, 30)
    else:
        result = cv2.cvtColor(gray_enhanced, cv2.COLOR_GRAY2RGB)
    
    return result, edges

def separate_black_sky_objects(image_np):
    """
    Special handling for images where sky is black and objects are dark.
    Uses texture and edge information to distinguish classes.
    """
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
    
    # Analyze texture using local standard deviation
    kernel_size = 5
    local_mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
    local_mean_sq = cv2.blur((gray.astype(np.float32) ** 2), (kernel_size, kernel_size))
    local_std = np.sqrt(local_mean_sq - local_mean ** 2 + 1e-8)
    
    # Normalize local std to 0-255
    local_std_norm = (local_std / local_std.max() * 255).astype(np.uint8)
    
    # Sky is typically smooth (low texture), objects have texture
    sky_mask = local_std_norm < 20
    object_mask = local_std_norm >= 20
    
    # Enhance differently for sky vs objects
    result = image_np.copy()
    
    if len(image_np.shape) == 3:
        # For sky regions: boost slightly to distinguish from pure black
        result[sky_mask] = cv2.addWeighted(result[sky_mask], 1.0, 
                                           np.full_like(result[sky_mask], 30), 0.3, 0)
        
        # For object regions: enhance contrast
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Enhance luminance for objects
        l_objects = l.copy()
        l_objects[object_mask] = cv2.addWeighted(l[object_mask], 1.5, 
                                                 l[object_mask], 0, 20)
        
        lab = cv2.merge([l_objects, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return result, local_std_norm

def synthetic_black_sky_augmentation(image_np, sky_class_mask=None):
    """
    Create synthetic training data with black sky.
    Simulates the test condition for training.
    """
    result = image_np.copy()
    h, w = image_np.shape[:2]
    
    # Create synthetic sky region (top 30% of image)
    sky_height = int(h * 0.3)
    
    if sky_class_mask is not None:
        # Use provided mask for sky
        sky_mask = sky_class_mask
    else:
        # Assume top region is sky
        sky_mask = np.zeros((h, w), dtype=bool)
        sky_mask[:sky_height, :] = True
    
    # Make sky black with slight variations
    if len(image_np.shape) == 3:
        # Create black sky with noise
        noise = np.random.randint(0, 15, (h, w, 3))
        result[sky_mask] = noise[sky_mask]
    
    return result

def preprocess_monochromatic_black(image_np):
    """
    Complete preprocessing pipeline for monochromatic black images.
    """
    # Analyze image
    analysis = analyze_black_content(image_np)
    
    print(f"  Black content: {analysis['black_percent']:.1f}%")
    print(f"  Dark content: {analysis['dark_percent']:.1f}%")
    print(f"  Monochromatic: {analysis['is_monochromatic']}")
    
    if analysis['is_monochromatic'] or analysis['dark_percent'] > 30:
        print("  → Applying monochromatic enhancement")
        
        # Apply specialized enhancement
        enhanced, edges = enhance_monochromatic_image(image_np)
        
        # Further separate sky/objects if needed
        if analysis['black_percent'] > 20:
            enhanced, texture = separate_black_sky_objects(enhanced)
        
        return enhanced, True  # True = was enhanced
    
    return image_np, False  # False = no enhancement needed

# ═══════════════════════════════════════════════════════════════
# TRAINING DATA GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_synthetic_black_training_data(image_np, mask_np, num_variants=3):
    """
    Generate synthetic training images with black sky variations.
    This helps the model learn to handle the test condition.
    """
    variants = []
    
    # Identify sky region from mask (class 9)
    sky_mask = (mask_np == 9)
    
    for i in range(num_variants):
        # Create variant with different black sky intensity
        variant = image_np.copy()
        
        if len(variant.shape) == 3:
            # Vary the blackness of sky
            intensity = np.random.randint(5, 30)
            noise = np.random.randint(0, intensity, variant.shape)
            
            # Apply to sky region
            variant[sky_mask] = noise[sky_mask]
            
            # Optionally darken objects slightly
            if i > 0:
                darken_factor = 0.7 + (i * 0.1)
                non_sky = ~sky_mask
                if len(variant.shape) == 3:
                    for c in range(3):
                        variant[:, :, c][non_sky] = (
                            variant[:, :, c][non_sky] * darken_factor
                        ).astype(np.uint8)
        
        variants.append(variant)
    
    return variants

# ═══════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test on a sample image
    test_paths = [
        "Offroad_Segmentation_testImages/Offroad_Segmentation_testImages/Color_Images/0000113.png",
        "Offroad_Segmentation_testImages/Offroad_Segmentation_testImages/Color_Images/0000060.png",
    ]
    
    for path in test_paths:
        if not os.path.exists(path):
            continue
            
        print(f"\n{'='*60}")
        print(f"Processing: {path}")
        print(f"{'='*60}")
        
        # Load image
        img = np.array(Image.open(path).convert('RGB'))
        
        # Analyze and enhance
        enhanced, was_enhanced = preprocess_monochromatic_black(img)
        
        # Save results
        output_dir = "black_enhancement_test"
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = Path(path).stem
        Image.fromarray(img).save(f"{output_dir}/{base_name}_original.png")
        Image.fromarray(enhanced).save(f"{output_dir}/{base_name}_enhanced.png")
        
        if was_enhanced:
            print(f"  ✓ Enhanced image saved to {output_dir}/{base_name}_enhanced.png")
    
    print(f"\n{'='*60}")
    print("TEST COMPLETE")
    print(f"{'='*60}")
    print(f"Check {output_dir}/ for before/after images")
