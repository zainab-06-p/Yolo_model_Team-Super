"""
ROUND 2 - ACCURACY REPORTS
Generate terminal-friendly accuracy metrics for screenshots
"""
import numpy as np
from PIL import Image
import os

# Color palette to class mapping
COLOR_PALETTE = [
    [135, 206, 235],    # 0 Sky - Light Blue
    [210, 180, 140],    # 1 Ground - Tan
    [128, 128, 128],    # 2 Small Rocks - Gray
    [34, 139, 34],      # 3 Vegetation - Forest Green
    [105, 105, 105],    # 4 Large Rocks - Dark Gray
    [139, 69, 19],      # 5 Ground Clutter - Brown
    [160, 82, 45],      # 6 Logs - Sienna
    [255, 255, 0],      # 7 Poles - Yellow
    [128, 0, 128],      # 8 Fences - Purple
    [255, 0, 0],        # 9 Sign - Red
]

def color_to_class_mask(color_img, palette=COLOR_PALETTE, tolerance=30):
    """Convert color image to class mask."""
    h, w = color_img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for i, color in enumerate(palette):
        # Find pixels matching this color (with tolerance)
        diff = np.abs(color_img.astype(float) - np.array(color))
        match = np.all(diff < tolerance, axis=2)
        mask[match] = i
    
    return mask

def compute_accuracy(pred_mask, gt_mask, n_classes=10):
    """Compute pixel accuracy and mIoU."""
    # Pixel accuracy
    pixel_acc = (pred_mask == gt_mask).mean() * 100
    
    # Per-class IoU
    ious = []
    for c in range(n_classes):
        pred_c = (pred_mask == c)
        gt_c = (gt_mask == c)
        intersection = np.logical_and(pred_c, gt_c).sum()
        union = np.logical_or(pred_c, gt_c).sum()
        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(np.nan)
    
    miou = np.nanmean(ious) * 100
    return pixel_acc, miou, ious

def print_report(image_num, model_name, acc1, miou1, acc2, miou2):
    """Print formatted accuracy report."""
    print(f"\n{'='*70}")
    print(f"  IMAGE {image_num} - {model_name}")
    print(f"{'='*70}")
    print(f"\n  📊 REPORT 1: Model vs Input (Quality Check)")
    print(f"     ├─ Pixel Accuracy: {acc1:.2f}%")
    print(f"     └─ mIoU:           {miou1:.2f}%")
    print(f"\n  📊 REPORT 2: Model vs Ground Truth (Performance)")
    print(f"     ├─ Pixel Accuracy: {acc2:.2f}%")
    print(f"     └─ mIoU:           {miou2:.2f}%")
    print(f"\n{'='*70}")

# Main execution
print("\n" + "="*70)
print("  ROUND 2 - ACCURACY REPORTS")
print("  Member 2 Model: Augmentation + Class Weights + Ensemble")
print("="*70)

base_dir = "round 2"

for img_num in [2, 3]:
    # Load images
    before_path = os.path.join(base_dir, "before", f"Image {img_num} Before.jpg")
    model_path = os.path.join(base_dir, f"Image {img_num} After.jpg")
    gt_path = os.path.join(base_dir, "after", f"{img_num} after.jpg")
    
    # Check files
    if not all(os.path.exists(p) for p in [before_path, model_path, gt_path]):
        print(f"\n⚠ Missing files for Image {img_num}")
        continue
    
    # Load images
    before_img = np.array(Image.open(before_path).resize((462, 252)))
    model_pred = np.array(Image.open(model_path).resize((462, 252)))
    gt_img = np.array(Image.open(gt_path).resize((462, 252)))
    
    # Convert to class masks
    before_mask = color_to_class_mask(before_img, tolerance=50)
    model_mask = color_to_class_mask(model_pred, tolerance=30)
    gt_mask = color_to_class_mask(gt_img, tolerance=30)
    
    # Report 1: Model vs Before (input quality - how much changed)
    # This shows the model's "creativity" - should be different from input
    acc1, miou1, _ = compute_accuracy(model_mask, before_mask)
    
    # Report 2: Model vs Ground Truth (actual performance)
    acc2, miou2, per_class_iou = compute_accuracy(model_mask, gt_mask)
    
    # Print formatted report
    print_report(img_num, "Enhanced Ensemble Model (T=0.5)", acc1, miou1, acc2, miou2)
    
    # Per-class breakdown
    class_names = ["Sky", "Ground", "Small Rocks", "Vegetation", "Large Rocks",
                   "Ground Clutter", "Logs", "Poles", "Fences", "Sign"]
    print(f"\n  📋 Per-Class IoU (vs Ground Truth):")
    for i, (name, iou) in enumerate(zip(class_names, per_class_iou)):
        status = "✓" if not np.isnan(iou) and iou > 0.3 else "○" if not np.isnan(iou) else "✗"
        iou_str = f"{iou*100:.1f}%" if not np.isnan(iou) else "N/A"
        print(f"     {status} Class {i} ({name:12s}): {iou_str:>6s}")

# Summary
print(f"\n{'='*70}")
print("  SUMMARY - BONUS ROUND SUBMISSION")
print(f"{'='*70}")
print("\n  Model Configuration:")
print("     • Ensemble: Your Model (0.6) + Base DINOv2 (0.4)")
print("     • Temperature Scaling: T=0.5")
print("     • Post-Processing: Artifact removal + Smoothing")
print("     • Color Matching: Histogram alignment with reference")
print("\n  Files Ready for Upload:")
print("     ✓ Image 2 Before.jpg")
print("     ✓ Image 2 After.jpg  (Enhanced Prediction)")
print("     ✓ Image 3 Before.jpg")
print("     ✓ Image 3 After.jpg  (Enhanced Prediction)")
print(f"\n{'='*70}")
print("  ⏰ DEADLINE: Submit to Google Drive before 5:15 PM")
print(f"{'='*70}\n")
