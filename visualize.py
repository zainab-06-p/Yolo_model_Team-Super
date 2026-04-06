"""
visualize.py — Visualization utilities for the DINOv2 Offroad Segmentation Ensemble

Generates:
  - Confusion matrix heatmap
  - Per-class IoU bar charts  
  - Failure case galleries (lowest IoU images)
  - Augmentation showcase
  - Ensemble comparison charts
  - Ablation study visualizations

Usage:
  python visualize.py --mode confusion --pred_dir predictions/masks --gt_dir val/Segmentation
  python visualize.py --mode failures --pred_dir predictions/masks --gt_dir val/Segmentation --n 10
  python visualize.py --mode per_class_iou --metrics_file evaluation_metrics.txt
  python visualize.py --mode ablation --metrics_file ablation_results.csv
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import cv2
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────
CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

RARE_CLASSES = [5, 6]  # Ground Clutter, Logs

COLOR_PALETTE = np.array([
    [0,   0,   0  ],   # 0 Background
    [34,  139, 34 ],   # 1 Trees
    [0,   255, 0  ],   # 2 Lush Bushes
    [210, 180, 140],    # 3 Dry Grass
    [139, 90,  43 ],    # 4 Dry Bushes
    [128, 128, 0  ],    # 5 Ground Clutter
    [139, 69,  19 ],     # 6 Logs
    [128, 128, 128],    # 7 Rocks
    [160, 82,  45 ],    # 8 Landscape
    [135, 206, 235],    # 9 Sky
], dtype=np.uint8)

VALUE_MAP = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

OUTPUT_DIR = "visualization_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def convert_mask_to_class_ids(mask_pil):
    """Convert 16-bit mask to class IDs 0-9."""
    arr = np.array(mask_pil)
    out = np.zeros_like(arr, dtype=np.uint8)
    for raw_val, class_id in VALUE_MAP.items():
        out[arr == raw_val] = class_id
    return out


def load_prediction_and_gt(pred_path, gt_path):
    """Load a prediction mask and its ground truth."""
    pred = np.array(Image.open(pred_path))
    gt_pil = Image.open(gt_path)
    gt = convert_mask_to_class_ids(gt_pil)
    return pred, gt


def compute_per_image_iou(pred, gt, n_classes=10):
    """Compute IoU for each class in a single image."""
    ious = []
    for c in range(n_classes):
        pred_c = pred == c
        gt_c = gt == c
        inter = np.logical_and(pred_c, gt_c).sum()
        union = np.logical_or(pred_c, gt_c).sum()
        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(inter / union)
    return ious


def compute_miou(pred, gt, n_classes=10):
    """Compute mean IoU."""
    ious = compute_per_image_iou(pred, gt, n_classes)
    return np.nanmean(ious), ious


# ── Visualization Functions ───────────────────────────────────

def plot_confusion_matrix(pred_dir, gt_dir, output_name="confusion_matrix.png"):
    """Generate confusion matrix heatmap from predictions vs ground truth."""
    print("Computing confusion matrix...")
    
    pred_files = sorted(Path(pred_dir).glob("*.png"))
    
    all_preds = []
    all_gts = []
    
    for pred_file in tqdm(pred_files, desc="Loading masks"):
        gt_file = Path(gt_dir) / pred_file.name.replace("_pred", "").replace("_color", "")
        if not gt_file.exists():
            gt_file = Path(gt_dir) / pred_file.name
        
        if gt_file.exists():
            pred, gt = load_prediction_and_gt(pred_file, gt_file)
            all_preds.extend(pred.flatten())
            all_gts.extend(gt.flatten())
    
    # Compute confusion matrix
    cm = confusion_matrix(all_gts, all_preds, labels=list(range(10)))
    
    # Normalize by row (ground truth)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('Ground Truth', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(axis='y', rotation=0)
    
    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax2, vmin=0, vmax=1, cbar_kws={'label': 'Proportion'})
    ax2.set_title('Confusion Matrix (Normalized by Ground Truth)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('Ground Truth', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, output_name)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved: {output_path}")
    return cm, cm_norm


def plot_per_class_iou(metrics_file=None, iou_values=None, output_name="per_class_iou.png", 
                         title="Per-Class IoU", member_names=None):
    """Generate bar chart of per-class IoU."""
    
    if metrics_file and os.path.exists(metrics_file):
        # Parse metrics file for IoU values
        iou_values = parse_metrics_file(metrics_file)
    
    if iou_values is None:
        print("Error: No IoU values provided or found")
        return
    
    # Handle single or multiple members
    if not isinstance(iou_values[0], (list, tuple, np.ndarray)):
        iou_values = [iou_values]
    
    n_members = len(iou_values)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(CLASS_NAMES))
    width = 0.8 / n_members
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_members))
    
    for i, (ious, color) in enumerate(zip(iou_values, colors)):
        label = member_names[i] if member_names else f"Model {i+1}"
        offset = (i - n_members/2 + 0.5) * width
        bar_vals = [0 if np.isnan(v) else v for v in ious]
        bar_colors = ['red' if v < 0.3 else 'orange' if v < 0.5 else 'green' 
                      for v in bar_vals]
        ax.bar(x + offset, bar_vals, width, label=label, 
               color=[c if i == 0 else color for c in bar_colors],
               alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('IoU', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add mean IoU line for first model
    mean_iou = np.nanmean(iou_values[0])
    ax.axhline(y=mean_iou, color='red', linestyle='--', linewidth=2,
               label=f'Mean IoU: {mean_iou:.4f}')
    
    # Highlight rare classes
    for rc in RARE_CLASSES:
        ax.axvspan(rc - 0.4, rc + 0.4, alpha=0.1, color='yellow', 
                   label='Rare class' if rc == RARE_CLASSES[0] else "")
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, output_name)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Per-class IoU chart saved: {output_path}")


def plot_failure_cases(pred_dir, gt_dir, img_dir, n_cases=5, output_name="failure_cases.png"):
    """Generate gallery of worst-performing images (lowest IoU)."""
    print(f"Finding top {n_cases} failure cases...")
    
    pred_files = sorted(Path(pred_dir).glob("*.png"))
    
    image_ious = []
    
    for pred_file in tqdm(pred_files, desc="Computing per-image IoU"):
        gt_file = Path(gt_dir) / pred_file.name.replace("_pred", "").replace("_color", "")
        if not gt_file.exists():
            gt_file = Path(gt_dir) / pred_file.name
        
        if gt_file.exists():
            pred, gt = load_prediction_and_gt(pred_file, gt_file)
            miou, _ = compute_miou(pred, gt)
            if not np.isnan(miou):
                image_ious.append((pred_file.name, miou, pred_file, gt_file))
    
    # Sort by IoU (ascending = worst first)
    image_ious.sort(key=lambda x: x[1])
    
    # Take worst n_cases
    worst_cases = image_ious[:n_cases]
    
    fig, axes = plt.subplots(n_cases, 3, figsize=(15, 5*n_cases))
    if n_cases == 1:
        axes = axes.reshape(1, -1)
    
    for i, (fname, miou, pred_file, gt_file) in enumerate(worst_cases):
        # Load original image
        img_file = Path(img_dir) / fname.replace("_pred", "").replace("_color", "")
        if img_file.exists():
            img = np.array(Image.open(img_file).convert('RGB'))
        else:
            img = np.zeros((252, 462, 3), dtype=np.uint8)
        
        pred, gt = load_prediction_and_gt(pred_file, gt_file)
        
        # Colorize masks
        pred_color = COLOR_PALETTE[pred]
        gt_color = COLOR_PALETTE[gt]
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Input: {fname}\n(mIoU: {miou:.4f})", fontsize=10)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gt_color)
        axes[i, 1].set_title("Ground Truth", fontsize=10)
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_color)
        axes[i, 2].set_title("Prediction", fontsize=10)
        axes[i, 2].axis('off')
    
    plt.suptitle(f"Top {n_cases} Failure Cases (Lowest mIoU)", 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, output_name)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Failure cases saved: {output_path}")


def plot_ablation_study(ablation_data, output_name="ablation_study.png"):
    """
    Generate ablation study visualization.
    
    ablation_data: dict with keys:
        'approaches': list of approach names
        'miou': list of mIoU values
        'improvement': list of improvements vs baseline
        'findings': list of key findings (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    approaches = ablation_data['approaches']
    mious = ablation_data['miou']
    improvements = ablation_data.get('improvement', [0]*len(approaches))
    
    colors = ['gray' if i == 0 else 'steelblue' for i in range(len(approaches))]
    colors[-1] = 'green'  # Final model in green
    
    # mIoU comparison
    bars1 = ax1.barh(approaches, mious, color=colors, edgecolor='black', alpha=0.85)
    ax1.set_xlabel('Validation mIoU', fontsize=12)
    ax1.set_title('Ablation Study: mIoU Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, mious)):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
    
    # Improvement chart
    bars2 = ax2.barh(approaches[1:], improvements[1:], color=colors[1:], 
                     edgecolor='black', alpha=0.85)
    ax2.set_xlabel('Improvement vs Baseline (%)', fontsize=12)
    ax2.set_title('Ablation Study: Improvement Over Baseline', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    for bar, val in zip(bars2, improvements[1:]):
        ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                f'+{val:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, output_name)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Ablation study saved: {output_path}")


def plot_training_curves(metrics_files, labels, output_name="training_curves.png"):
    """Plot training curves from multiple experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for metrics_file, label in zip(metrics_files, labels):
        if not os.path.exists(metrics_file):
            print(f"Warning: {metrics_file} not found")
            continue
        
        # Parse CSV or text file for epoch, loss, miou
        epochs, losses, mious = parse_training_log(metrics_file)
        
        axes[0].plot(epochs, losses, label=label, linewidth=2)
        axes[1].plot(epochs, mious, label=label, linewidth=2)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation mIoU')
    axes[1].set_title('Validation mIoU')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, output_name)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves saved: {output_path}")


def plot_augmentation_showcase(original_img_path, transform, n_variants=6, 
                                output_name="augmentation_showcase.png"):
    """Show original image with multiple augmented variants."""
    from PIL import Image
    import torchvision.transforms as T
    
    img = Image.open(original_img_path).convert('RGB')
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # Original
    axes[0].imshow(img)
    axes[0].set_title("Original", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Augmented variants
    for i in range(1, n_variants + 1):
        aug_img = transform(img)
        if torch.is_tensor(aug_img):
            aug_img = aug_img.permute(1, 2, 0).numpy()
            aug_img = aug_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            aug_img = np.clip(aug_img, 0, 1)
        axes[i].imshow(aug_img)
        axes[i].set_title(f"Augmentation {i}", fontsize=12)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_variants + 1, 8):
        axes[i].axis('off')
    
    plt.suptitle("Augmentation Showcase", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, output_name)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Augmentation showcase saved: {output_path}")


# ── Helper Functions ──────────────────────────────────────────

def parse_metrics_file(filepath):
    """Parse metrics file to extract IoU values."""
    iou_values = []
    with open(filepath, 'r') as f:
        for line in f:
            # Look for per-class IoU lines
            if ':' in line and any(c in line for c in CLASS_NAMES):
                try:
                    val = float(line.split(':')[1].strip().split()[0])
                    iou_values.append(val)
                except:
                    pass
    return iou_values if iou_values else None


def parse_training_log(filepath):
    """Parse training log for epochs, losses, and mIoU."""
    epochs, losses, mious = [], [], []
    with open(filepath, 'r') as f:
        for line in f:
            # Try to parse epoch data
            if 'epoch' in line.lower() or 'ep' in line.lower():
                try:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p.isdigit():
                            epochs.append(int(p))
                        if 'loss' in p.lower() and i+1 < len(parts):
                            try:
                                losses.append(float(parts[i+1]))
                            except:
                                pass
                        if 'miou' in p.lower() or 'iou' in p.lower():
                            try:
                                mious.append(float(parts[i+1]))
                            except:
                                pass
                except:
                    pass
    return epochs, losses, mious


# ── Main Entry Point ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualization tools for segmentation results')
    parser.add_argument('--mode', choices=['confusion', 'per_class_iou', 'failures', 
                                            'ablation', 'curves', 'augmentation'],
                        required=True, help='Visualization type')
    parser.add_argument('--pred_dir', help='Directory with prediction masks')
    parser.add_argument('--gt_dir', help='Directory with ground truth masks')
    parser.add_argument('--img_dir', help='Directory with original images')
    parser.add_argument('--metrics_file', help='Path to metrics file')
    parser.add_argument('--output', help='Output filename')
    parser.add_argument('--n', type=int, default=5, help='Number of failure cases')
    
    args = parser.parse_args()
    
    if args.mode == 'confusion':
        if not args.pred_dir or not args.gt_dir:
            print("Error: --pred_dir and --gt_dir required for confusion matrix")
            return
        plot_confusion_matrix(args.pred_dir, args.gt_dir, args.output or "confusion_matrix.png")
    
    elif args.mode == 'per_class_iou':
        plot_per_class_iou(args.metrics_file, output_name=args.output or "per_class_iou.png")
    
    elif args.mode == 'failures':
        if not args.pred_dir or not args.gt_dir or not args.img_dir:
            print("Error: --pred_dir, --gt_dir, and --img_dir required for failure cases")
            return
        plot_failure_cases(args.pred_dir, args.gt_dir, args.img_dir, args.n, 
                          args.output or "failure_cases.png")
    
    elif args.mode == 'ablation':
        print("Ablation mode requires manual data input. Use plot_ablation_study() programmatically.")
    
    elif args.mode == 'curves':
        print("Curves mode requires multiple metrics files. Use plot_training_curves() programmatically.")
    
    elif args.mode == 'augmentation':
        print("Augmentation mode requires original image path. Use plot_augmentation_showcase() programmatically.")
    
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
