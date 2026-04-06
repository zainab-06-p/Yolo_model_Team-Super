"""
Test ensemble model on images with ground truth masks
Computes mIoU for each image and reports best results
"""

import os
import sys
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Model paths (update these to your actual paths)
MEMBER1_PATH = 'submission/models/model_finetuned_best.pth.zip'
MEMBER2_PATH = 'submission/models/model_augmented_best.pth'
MEMBER3_PATH = 'submission/models/model_best.pth.zip'

# Test data paths - using validation data which has ground truth masks
TEST_IMAGES_DIR = 'desert-kaggle-api/val/Color_Images'
TEST_MASKS_DIR = 'desert-kaggle-api/val/Segmentation'

# Image settings
IMG_H, IMG_W = 252, 462
TOKEN_H, TOKEN_W = 18, 33
N_CLASSES = 10
N_EMB = 384

# Class mapping
VALUE_MAP = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

COLOR_PALETTE = np.array([
    [0, 0, 0], [34, 139, 34], [0, 255, 0], [210, 180, 140],
    [139, 90, 43], [128, 128, 0], [139, 69, 19], [128, 128, 128],
    [160, 82, 45], [135, 206, 235]
], dtype=np.uint8)

# ═══════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH, dropout=0.1):
        super().__init__()
        self.tokenH, self.tokenW = tokenH, tokenW
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 256, 7, padding=3), nn.GELU())
        self.block1 = nn.Sequential(
            nn.Conv2d(256, 256, 7, padding=3, groups=256), nn.GELU(),
            nn.Conv2d(256, 256, 1), nn.GELU(),
        )
        self.block2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.GELU())
        self.dropout = nn.Dropout2d(p=dropout)
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.tokenH, self.tokenW, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.dropout(x)
        return self.classifier(x)

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def extract_zip(zip_path):
    """Extract zip and return path to first .pth file."""
    if not zip_path.endswith('.zip'):
        return zip_path
    extract_dir = zip_path.replace('.zip', '_extracted')
    if not os.path.exists(extract_dir):
        print(f"Extracting {zip_path}...")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
    pth_files = list(Path(extract_dir).rglob('*.pth'))
    return str(pth_files[0]) if pth_files else zip_path

def load_head(ckpt_path, name):
    """Load model head from checkpoint."""
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print(f"⚠ {name}: Not found at {ckpt_path}")
        return None
    
    resolved = extract_zip(ckpt_path)
    
    head = SegmentationHeadConvNeXt(N_EMB, N_CLASSES, TOKEN_W, TOKEN_H).to(device)
    
    try:
        state = torch.load(resolved, map_location=device, weights_only=False)
        
        if isinstance(state, dict):
            if 'classifier_state' in state:
                state = state['classifier_state']
            elif 'state_dict' in state:
                state = state['state_dict']
            elif 'model' in state:
                state = state['model']
        elif isinstance(state, nn.Module):
            state = state.state_dict()
        
        head.load_state_dict(state, strict=True)
        head.eval()
        print(f"✓ {name}: Loaded from {ckpt_path}")
        return head
    except Exception as e:
        print(f"⚠ {name}: Load failed - {e}")
        return None

# Transforms
img_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def convert_mask(mask_pil):
    """Convert 16-bit mask to class IDs."""
    arr = np.array(mask_pil)
    out = np.zeros_like(arr, dtype=np.uint8)
    for raw, cls in VALUE_MAP.items():
        out[arr == raw] = cls
    return out

def compute_iou(pred, target, n_classes=10):
    """Compute mean IoU and per-class IoU."""
    ious = []
    for c in range(n_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = np.logical_and(pred_c, target_c).sum()
        union = np.logical_or(pred_c, target_c).sum()
        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(intersection / union)
    return np.nanmean(ious), ious

# ═══════════════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("LOADING ENSEMBLE MODELS")
print("="*60)

# Load backbone
print("Loading DINOv2 backbone...")
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', 
                          pretrained=True, verbose=False)
backbone = backbone.to(device).eval()
for p in backbone.parameters():
    p.requires_grad = False
print("✓ Backbone loaded")

# Load heads
heads = []
m1 = load_head(MEMBER1_PATH, "Member 1 (Fine-tune)")
if m1:
    heads.append(("M1", m1, 0.40))
    
m2 = load_head(MEMBER2_PATH, "Member 2 (Augmented)")
if m2:
    heads.append(("M2", m2, 0.30))
    
m3 = load_head(MEMBER3_PATH, "Member 3 (Hyperparams)")
if m3:
    heads.append(("M3", m3, 0.30))

if not heads:
    raise RuntimeError("No models loaded! Check paths.")

print(f"\n✓ Loaded {len(heads)} model(s)")
for name, _, w in heads:
    print(f"  - {name}: weight={w}")

# ═══════════════════════════════════════════════════════════════
# INFERENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_single(img_tensor, head):
    """Single model prediction."""
    img_batch = img_tensor.unsqueeze(0).to(device)
    feats = backbone.forward_features(img_batch)['x_norm_patchtokens']
    logits = head(feats)
    logits = F.interpolate(logits, size=(IMG_H, IMG_W), mode='bilinear', align_corners=False)
    return F.softmax(logits, dim=1).squeeze(0)

@torch.no_grad()
def predict_ensemble(img_tensor, use_tta=True):
    """Ensemble prediction with optional TTA."""
    img_tensor = img_tensor.to(device)
    
    total_weight = sum(w for _, _, w in heads)
    combined = torch.zeros(N_CLASSES, IMG_H, IMG_W, device=device)
    
    for name, head, weight in heads:
        # Original
        probs = predict_single(img_tensor.cpu(), head)
        
        if use_tta:
            # Horizontal flip TTA
            img_flip = torch.flip(img_tensor, dims=[-1])
            probs_flip = predict_single(img_flip.cpu(), head)
            probs_flip = torch.flip(probs_flip, dims=[-1])
            probs = (probs + probs_flip.cpu()) / 2
        else:
            probs = probs.cpu()
        
        combined += weight * probs.to(device)
    
    combined /= total_weight
    return torch.argmax(combined, dim=0).cpu().numpy()

# ═══════════════════════════════════════════════════════════════
# TEST ON IMAGES WITH GROUND TRUTH
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TESTING ON VALIDATION IMAGES (with ground truth)")
print("="*60)

# Check directories
if not os.path.exists(TEST_IMAGES_DIR):
    print(f"⚠ Images not found: {TEST_IMAGES_DIR}")
    sys.exit(1)
    
if not os.path.exists(TEST_MASKS_DIR):
    print(f"⚠ Masks not found: {TEST_MASKS_DIR}")
    sys.exit(1)

# Get image files
image_files = sorted([f for f in os.listdir(TEST_IMAGES_DIR) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

print(f"\nFound {len(image_files)} validation images")

# Test on images
N_TEST = min(20, len(image_files))  # Test on up to 20 images
test_files = image_files[:N_TEST]

print(f"\nTesting on {N_TEST} images...")

results = []
all_mious = []

for fname in tqdm(test_files, desc="Processing"):
    img_path = os.path.join(TEST_IMAGES_DIR, fname)
    mask_path = os.path.join(TEST_MASKS_DIR, fname)
    
    # Check if mask exists
    if not os.path.exists(mask_path):
        print(f"⚠ Mask not found: {mask_path}, skipping...")
        continue
    
    # Load image and mask
    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path)
    mask_np = convert_mask(mask)
    
    # Resize mask to match prediction size
    mask_resized = Image.fromarray(mask_np).resize((IMG_W, IMG_H), Image.NEAREST)
    mask_np_resized = np.array(mask_resized)
    
    # Preprocess image
    img_tensor = img_transform(img)
    
    # Run ensemble inference
    pred = predict_ensemble(img_tensor, use_tta=True)
    
    # Compute mIoU
    miou, class_ious = compute_iou(pred, mask_np_resized)
    all_mious.append(miou)
    
    # Save outputs
    output_dir = 'test_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = Path(fname).stem
    
    # Save prediction
    Image.fromarray(pred.astype(np.uint8)).save(f"{output_dir}/{base_name}_pred.png")
    
    # Save colorized prediction
    color_pred = COLOR_PALETTE[pred]
    cv2.imwrite(f"{output_dir}/{base_name}_pred_color.png", 
                cv2.cvtColor(color_pred, cv2.COLOR_RGB2BGR))
    
    # Save ground truth colorized
    color_gt = COLOR_PALETTE[mask_np_resized]
    cv2.imwrite(f"{output_dir}/{base_name}_gt_color.png", 
                cv2.cvtColor(color_gt, cv2.COLOR_RGB2BGR))
    
    # Create comparison
    img_np = np.array(img.resize((IMG_W, IMG_H)))
    comparison = np.hstack([img_np, color_pred, color_gt])
    cv2.imwrite(f"{output_dir}/{base_name}_compare.png", 
                cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    # Count predicted classes
    unique, counts = np.unique(pred, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    
    results.append({
        'filename': fname,
        'miou': miou,
        'class_ious': class_ious,
        'prediction': pred,
        'class_distribution': class_distribution,
        'num_classes': len(unique)
    })

# ═══════════════════════════════════════════════════════════════
# ANALYSIS & REPORT
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

print(f"\nProcessed {len(results)} images")
print(f"Output saved to: test_outputs/")

if all_mious:
    mean_miou = np.nanmean(all_mious)
    print(f"\n{'='*60}")
    print(f"ENSEMBLE mIoU: {mean_miou:.4f}")
    print(f"{'='*60}")

# Sort by mIoU
results_sorted = sorted(results, key=lambda x: x['miou'], reverse=True)

print("\n" + "-"*60)
print("TOP 5 IMAGES BY mIoU:")
print("-"*60)
for i, r in enumerate(results_sorted[:5], 1):
    print(f"\n{i}. {r['filename']}")
    print(f"   mIoU: {r['miou']:.4f}")
    print(f"   Classes detected: {r['num_classes']}")
    
    # Show per-class IoU
    print("   Per-class IoU:")
    for cls_id, iou in enumerate(r['class_ious']):
        if not np.isnan(iou) and iou > 0:
            print(f"     - {CLASS_NAMES[cls_id]}: {iou:.4f}")

print("\n" + "-"*60)
print("BOTTOM 5 IMAGES BY mIoU:")
print("-"*60)
for i, r in enumerate(results_sorted[-5:], 1):
    print(f"\n{i}. {r['filename']}")
    print(f"   mIoU: {r['miou']:.4f}")
    print(f"   Classes detected: {r['num_classes']}")

# Best overall
best = results_sorted[0]
print(f"\n{'='*60}")
print("BEST IMAGE OVERALL")
print(f"{'='*60}")
print(f"File: {best['filename']}")
print(f"mIoU: {best['miou']:.4f}")
print(f"Classes: {best['num_classes']}")
print("\nPer-class breakdown:")
for cls_id, iou in enumerate(best['class_ious']):
    status = "✓" if not np.isnan(iou) and iou > 0.5 else "○" if not np.isnan(iou) else "✗"
    iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
    print(f"  {status} {CLASS_NAMES[cls_id]:<16}: {iou_str}")

# Per-class average IoU across all images
print(f"\n{'='*60}")
print("PER-CLASS AVERAGE IOU (all images)")
print(f"{'='*60}")
class_ious_all = np.array([r['class_ious'] for r in results])
mean_class_ious = np.nanmean(class_ious_all, axis=0)

for cls_id, iou in enumerate(mean_class_ious):
    iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
    bar = "█" * int(iou * 20) if not np.isnan(iou) else ""
    print(f"  {CLASS_NAMES[cls_id]:<16}: {iou_str} {bar}")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
print(f"\nOutputs saved in: test_outputs/")
print("  - *_pred.png      : Raw prediction (class IDs)")
print("  - *_pred_color.png: Colorized prediction")
print("  - *_gt_color.png  : Colorized ground truth")
print("  - *_compare.png   : Side-by-side comparison")
print(f"\nBest mIoU achieved: {results_sorted[0]['miou']:.4f}")
print(f"Mean mIoU across {len(results)} images: {mean_miou:.4f}")
