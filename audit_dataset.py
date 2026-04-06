"""
╔══════════════════════════════════════════════════════════════════╗
║  TASK 1 — audit_dataset.py                                       ║
║  Full dataset audit: class distribution, weights, charts          ║
║  Member 2 — Duality AI Hackathon                                  ║
║                                                                    ║
║  Usage: python audit_dataset.py                                   ║
║  Or paste cells into Kaggle notebook                               ║
╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# ================================================================
# CONFIGURATION — Update paths for your environment
# ================================================================

# LOCAL PATHS (Windows)
TRAIN_DIR = r'Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train'
VAL_DIR   = r'Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\val'
TEST_DIR  = r'Offroad_Segmentation_testImages\Offroad_Segmentation_testImages'

# KAGGLE PATHS (uncomment when running on Kaggle)
# TRAIN_DIR = '/kaggle/input/YOUR-DATASET/Offroad_Segmentation_Training_Dataset/train'
# VAL_DIR   = '/kaggle/input/YOUR-DATASET/Offroad_Segmentation_Training_Dataset/val'
# TEST_DIR  = '/kaggle/input/YOUR-DATASET/Offroad_Segmentation_testImages'

OUTPUT_DIR = 'audit_results'

# ================================================================
# CLASS MAPPING (from provided scripts — DO NOT CHANGE)
# ================================================================
VALUE_MAP = {
    0: 0,        # Background (may not exist in data)
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    600: 6,      # Flowers (in data but not in provided script!)
    700: 7,      # Logs (was index 6 in provided script)
    800: 8,      # Rocks
    7100: 9,     # Landscape
    10000: 10,   # Sky
}

# The PROVIDED script uses this mapping (10 classes, 0-9):
VALUE_MAP_PROVIDED = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

CLASS_NAMES_PROVIDED = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

N_CLASSES = 10  # Matching the provided script

# ================================================================
# CORE FUNCTIONS
# ================================================================

def scan_masks(mask_dir, sample_limit=None):
    """Scan all masks in a directory. Returns per-class pixel counts and image counts."""
    files = sorted(os.listdir(mask_dir))
    if sample_limit:
        files = files[:sample_limit]
    
    pixel_counts = {}    # raw_value -> total pixel count
    image_counts = {}    # raw_value -> number of images containing it
    image_resolutions = set()
    
    for fname in tqdm(files, desc=f"Scanning {os.path.basename(os.path.dirname(mask_dir))}"):
        mask_pil = Image.open(os.path.join(mask_dir, fname))
        image_resolutions.add(mask_pil.size)
        mask = np.array(mask_pil)
        
        unique_vals = np.unique(mask)
        for v in unique_vals:
            v = int(v)
            count = int((mask == v).sum())
            pixel_counts[v] = pixel_counts.get(v, 0) + count
            image_counts[v] = image_counts.get(v, 0) + 1
    
    return pixel_counts, image_counts, image_resolutions, len(files)


def compute_class_weights(pixel_counts, n_classes=10, method='inv_sqrt'):
    """
    Compute class weights from pixel counts.
    
    Methods:
    - 'inv_sqrt': 1/sqrt(freq) — dampened inverse frequency
    - 'inv_freq': 1/freq — raw inverse frequency (can be extreme)
    - 'effective': effective number of samples (Cui et al., 2019)
    """
    # Convert raw pixel values to class IDs using the provided script's mapping
    class_pixels = np.zeros(n_classes, dtype=np.float64)
    for raw_val, class_id in VALUE_MAP_PROVIDED.items():
        class_pixels[class_id] = pixel_counts.get(raw_val, 0)
    
    total = class_pixels.sum()
    freqs = class_pixels / (total + 1e-8)
    
    if method == 'inv_sqrt':
        weights = 1.0 / np.sqrt(freqs + 1e-4)
    elif method == 'inv_freq':
        weights = 1.0 / (freqs + 1e-4)
    elif method == 'effective':
        beta = 0.9999
        weights = (1 - beta) / (1 - np.power(beta, class_pixels + 1e-4))
    
    # Normalize so weights sum = n_classes
    weights = weights / weights.sum() * n_classes
    
    return weights, freqs, class_pixels


def print_distribution(pixel_counts, image_counts, n_images, split_name):
    """Pretty-print class distribution."""
    total_pixels = sum(pixel_counts.values())
    
    print(f"\n{'='*75}")
    print(f"  {split_name} SPLIT — {n_images} images")
    print(f"{'='*75}")
    print(f"{'Raw ID':>8} | {'Name':<16} | {'Pixels':>14} | {'%':>7} | {'In N Images':>11}")
    print(f"{'-'*75}")
    
    for raw_val in sorted(pixel_counts.keys()):
        pct = pixel_counts[raw_val] / total_pixels * 100
        n_imgs = image_counts.get(raw_val, 0)
        
        # Look up name
        if raw_val in VALUE_MAP_PROVIDED:
            name = CLASS_NAMES_PROVIDED[VALUE_MAP_PROVIDED[raw_val]]
        elif raw_val == 600:
            name = "Flowers (!)"
        else:
            name = f"UNKNOWN_{raw_val}"
        
        # Severity indicator
        severity = ""
        if pct < 0.1:
            severity = " 🚨 CRITICAL"
        elif pct < 1.0:
            severity = " ⚠️  RARE"
        elif pct < 3.0:
            severity = " ⚡ UNCOMMON"
        
        print(f"{raw_val:>8} | {name:<16} | {pixel_counts[raw_val]:>14,} | {pct:>6.2f}% | {n_imgs:>5}/{n_images}{severity}")
    
    return total_pixels


def generate_charts(train_freqs, weights, class_names, output_dir):
    """Generate publication-quality charts for the report."""
    os.makedirs(output_dir, exist_ok=True)
    
    # ── Chart 1: Class Distribution ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color code: red=critical, orange=rare, blue=normal
    colors = []
    for f in train_freqs:
        if f < 0.001:
            colors.append('#FF4444')  # critical
        elif f < 0.01:
            colors.append('#FF8844')  # rare
        elif f < 0.05:
            colors.append('#FFAA44')  # uncommon
        else:
            colors.append('#4488CC')  # normal
    
    bars = axes[0].bar(range(len(class_names)), train_freqs * 100, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(range(len(class_names)))
    axes[0].set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    axes[0].set_ylabel('Pixel Frequency (%)', fontsize=11)
    axes[0].set_title('Class Pixel Distribution (Training Set)', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for bar, freq in zip(bars, train_freqs):
        if freq * 100 > 0.5:
            axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                        f'{freq*100:.1f}%', ha='center', va='bottom', fontsize=8)
        else:
            axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                        f'{freq*100:.2f}%', ha='center', va='bottom', fontsize=7, color='red')
    
    # Class weights
    axes[1].bar(range(len(class_names)), weights, color='mediumpurple', edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(range(len(class_names)))
    axes[1].set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    axes[1].set_ylabel('Weight Multiplier', fontsize=11)
    axes[1].set_title('Computed Class Weights (inv-sqrt)', fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline (1.0)')
    axes[1].legend()
    
    for i, w in enumerate(weights):
        axes[1].text(i, w + 0.05, f'{w:.1f}x', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/class_distribution.png")
    
    # ── Chart 2: Train vs Test comparison ──
    return True


def generate_train_vs_test_chart(train_pixel_counts, test_pixel_counts, class_names, output_dir):
    """Compare train and test class distributions side by side."""
    train_total = sum(train_pixel_counts.values())
    test_total = sum(test_pixel_counts.values())
    
    train_pcts = []
    test_pcts = []
    for raw_val in [0, 100, 200, 300, 500, 550, 700, 800, 7100, 10000]:
        train_pcts.append(train_pixel_counts.get(raw_val, 0) / train_total * 100)
        test_pcts.append(test_pixel_counts.get(raw_val, 0) / test_total * 100)
    
    x = np.arange(len(class_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, train_pcts, width, label='Train', color='#4488CC', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, test_pcts, width, label='Test', color='#CC4444', edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Pixel %', fontsize=12)
    ax.set_title('Train vs Test: Class Distribution Shift', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Highlight classes that are MISSING from test
    for i, (tp, tep) in enumerate(zip(train_pcts, test_pcts)):
        if tp > 0.5 and tep < 0.01:
            ax.annotate('MISSING\nin test!', xy=(i + width/2, 0), fontsize=7,
                       color='red', fontweight='bold', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'train_vs_test_distribution.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/train_vs_test_distribution.png")


def save_results(weights, freqs, class_names, pixel_counts_all, output_dir):
    """Save weights as .pt and full results as .txt."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save PyTorch tensor
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    torch.save(weights_tensor, os.path.join(output_dir, 'class_weights.pt'))
    print(f"✓ Saved: {output_dir}/class_weights.pt")
    print(f"  → Load with: class_weights = torch.load('class_weights.pt')")
    print(f"  → Values: {weights_tensor.tolist()}")
    
    # Save text report
    with open(os.path.join(output_dir, 'dataset_audit_results.txt'), 'w') as f:
        f.write("DUALITY AI HACKATHON — DATASET AUDIT RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("CLASS WEIGHTS (inv-sqrt method):\n")
        f.write("-" * 40 + "\n")
        for i, (name, w, freq) in enumerate(zip(class_names, weights, freqs)):
            f.write(f"  [{i}] {name:<16}: weight={w:.4f}, freq={freq*100:.3f}%\n")
        
        f.write(f"\nPyTorch tensor:\n")
        f.write(f"  torch.tensor({weights.tolist()})\n\n")
        
        f.write("RAW PIXEL COUNTS (all splits):\n")
        f.write("-" * 40 + "\n")
        for split_name, pc in pixel_counts_all.items():
            f.write(f"\n{split_name}:\n")
            for raw_val in sorted(pc.keys()):
                f.write(f"  {raw_val:>6}: {pc[raw_val]:>14,} pixels\n")
    
    print(f"✓ Saved: {output_dir}/dataset_audit_results.txt")


# ================================================================
# MAIN
# ================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("╔══════════════════════════════════════════════════╗")
    print("║  DATASET AUDIT — Duality AI Hackathon             ║")
    print("╚══════════════════════════════════════════════════╝")
    
    # ── Scan all splits ──
    print("\n[1/3] Scanning TRAINING set (all masks)...")
    train_pc, train_ic, train_res, train_n = scan_masks(os.path.join(TRAIN_DIR, 'Segmentation'))
    print_distribution(train_pc, train_ic, train_n, "TRAIN")
    
    print("\n[2/3] Scanning VALIDATION set...")
    val_pc, val_ic, val_res, val_n = scan_masks(os.path.join(VAL_DIR, 'Segmentation'))
    print_distribution(val_pc, val_ic, val_n, "VAL")
    
    print("\n[3/3] Scanning TEST set...")
    test_pc, test_ic, test_res, test_n = scan_masks(os.path.join(TEST_DIR, 'Segmentation'))
    print_distribution(test_pc, test_ic, test_n, "TEST")
    
    # ── Check for Flowers (class 600) ──
    print("\n" + "=" * 75)
    print("  ⚠️  CLASS MAPPING VERIFICATION")
    print("=" * 75)
    
    has_flowers = 600 in train_pc
    has_background = 0 in train_pc
    
    if has_flowers:
        pct = train_pc[600] / sum(train_pc.values()) * 100
        print(f"  ✅ Flowers (raw=600) FOUND in training data: {pct:.2f}% of pixels")
        print(f"     BUT the provided script does NOT map 600!")
        print(f"     → Pixels with value 600 will be mapped to class 0 (Background)")
        print(f"     → This means Flowers are being SILENTLY IGNORED")
    else:
        print(f"  ❌ Flowers (raw=600) NOT found in data")
    
    if has_background:
        print(f"  ✅ Background (raw=0) found in data: {train_pc[0]:,} pixels")
    else:
        print(f"  ❌ Background (raw=0) NOT found in data — class 0 is phantom")
    
    # ── Test set domain shift analysis ──
    print(f"\n  🔍 TEST SET DOMAIN SHIFT:")
    train_classes = set(train_pc.keys())
    test_classes = set(test_pc.keys())
    missing = train_classes - test_classes
    new_in_test = test_classes - train_classes
    
    if missing:
        print(f"  Classes in TRAIN but NOT in TEST: {sorted(missing)}")
        for v in sorted(missing):
            if v in VALUE_MAP_PROVIDED:
                name = CLASS_NAMES_PROVIDED[VALUE_MAP_PROVIDED[v]]
            elif v == 600:
                name = "Flowers"
            else:
                name = f"Unknown_{v}"
            print(f"    → {v} ({name})")
    
    if new_in_test:
        print(f"  Classes in TEST but NOT in TRAIN: {sorted(new_in_test)}")
    
    # ── Compute class weights ──
    print("\n" + "=" * 75)
    print("  COMPUTING CLASS WEIGHTS")
    print("=" * 75)
    
    weights, freqs, class_pixels = compute_class_weights(train_pc, N_CLASSES, method='inv_sqrt')
    
    print(f"\nClass weights (using provided script's VALUE_MAP with {N_CLASSES} classes):")
    for i, (name, w, f) in enumerate(zip(CLASS_NAMES_PROVIDED, weights, freqs)):
        flag = ""
        if w > 3.0:
            flag = " ← RARE (high weight)"
        elif w < 0.8:
            flag = " ← DOMINANT (low weight)"
        print(f"  [{i}] {name:<16}: weight={w:.4f}  freq={f*100:.3f}%{flag}")
    
    # ── Generate charts ──
    print("\nGenerating charts...")
    generate_charts(freqs, weights, CLASS_NAMES_PROVIDED, OUTPUT_DIR)
    generate_train_vs_test_chart(train_pc, test_pc, CLASS_NAMES_PROVIDED, OUTPUT_DIR)
    
    # ── Save results ──
    print("\nSaving results...")
    save_results(
        weights, freqs, CLASS_NAMES_PROVIDED,
        {'train': train_pc, 'val': val_pc, 'test': test_pc},
        OUTPUT_DIR
    )
    
    # ── Summary ──
    print("\n" + "=" * 75)
    print("  ✅ AUDIT COMPLETE")
    print("=" * 75)
    print(f"  Train: {train_n} images | Val: {val_n} images | Test: {test_n} images")
    print(f"  Resolutions: Train={train_res} | Val={val_res} | Test={test_res}")
    print(f"  Total classes in train: {len(train_pc)} raw values")
    print(f"  Total classes in test: {len(test_pc)} raw values")
    print(f"\n  Output files in '{OUTPUT_DIR}/':")
    print(f"    class_weights.pt              ← SHARE WITH TEAM")
    print(f"    class_distribution.png        ← FOR REPORT")
    print(f"    train_vs_test_distribution.png ← FOR REPORT")
    print(f"    dataset_audit_results.txt     ← FULL RESULTS")
    
    # Print the tensor in copy-paste format
    print(f"\n  📋 COPY-PASTE this into your training script:")
    print(f"  class_weights = torch.tensor({weights.tolist()}).to(device)")


if __name__ == '__main__':
    main()
