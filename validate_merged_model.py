"""
VALIDATION SCRIPT - Test Merged Model mIoU
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Paths
MERGED_MODEL_PATH = 'model_merged_final.pth'
VAL_DIR = 'Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/val'

# Settings
IMG_H, IMG_W = 252, 462
TOKEN_H, TOKEN_W = 18, 33
N_CLASSES = 10
N_EMB = 384
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class mapping
VALUE_MAP = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

INV_VALUE_MAP = {v: k for k, v in VALUE_MAP.items()}

# Class names
CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

print(f"Device: {DEVICE}")
print(f"Testing merged model: {MERGED_MODEL_PATH}")

# ═══════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.tokenH, self.tokenW = tokenH, tokenW
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 256, 7, padding=3), nn.GELU())
        self.block1 = nn.Sequential(
            nn.Conv2d(256, 256, 7, padding=3, groups=256), nn.GELU(),
            nn.Conv2d(256, 256, 1), nn.GELU(),
        )
        self.block2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.GELU())
        self.dropout = nn.Dropout2d(p=0.1)
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
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*50)
print("LOADING MODELS")
print("="*50)

# Load backbone
print("Loading DINOv2...")
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', 
                          pretrained=True, verbose=False)
backbone = backbone.to(DEVICE).eval()
for p in backbone.parameters():
    p.requires_grad = False
print("✓ Backbone loaded")

# Load merged head
print(f"Loading merged model from {MERGED_MODEL_PATH}...")
merged_head = SegmentationHeadConvNeXt(N_EMB, N_CLASSES, TOKEN_W, TOKEN_H).to(DEVICE)
state = torch.load(MERGED_MODEL_PATH, map_location=DEVICE, weights_only=False)
merged_head.load_state_dict(state, strict=True)
merged_head.eval()
print(f"✓ Merged model loaded ({len(state)} tensors)")

# ═══════════════════════════════════════════════════════════════
# TRANSFORMS
# ═══════════════════════════════════════════════════════════════

img_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def convert_mask_to_classes(mask_pil):
    """Convert mask PIL to class IDs"""
    arr = np.array(mask_pil)
    out = np.zeros_like(arr, dtype=np.uint8)
    for raw, cls in VALUE_MAP.items():
        out[arr == raw] = cls
    return out

# ═══════════════════════════════════════════════════════════════
# INFERENCE FUNCTION
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_merged(img_tensor):
    """Single merged model prediction with TTA"""
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    
    # Original
    feats = backbone.forward_features(img_tensor)['x_norm_patchtokens']
    logits = merged_head(feats)
    logits = torch.nn.functional.interpolate(logits, size=(IMG_H, IMG_W), 
                                              mode='bilinear', align_corners=False)
    probs = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
    
    # TTA: Horizontal flip
    img_flip = torch.flip(img_tensor, dims=[-1])
    feats_flip = backbone.forward_features(img_flip)['x_norm_patchtokens']
    logits_flip = merged_head(feats_flip)
    logits_flip = torch.nn.functional.interpolate(logits_flip, size=(IMG_H, IMG_W),
                                                    mode='bilinear', align_corners=False)
    probs_flip = torch.nn.functional.softmax(logits_flip, dim=1).squeeze().cpu().numpy()
    probs_flip = np.flip(probs_flip, axis=-1)
    
    # Average
    final_probs = (probs + probs_flip) / 2
    pred = np.argmax(final_probs, axis=0)
    
    return pred

# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════

def compute_iou_per_class(pred, target, n_classes):
    """Compute IoU for each class"""
    ious = []
    for c in range(n_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        inter = np.logical_and(pred_c, target_c).sum()
        union = np.logical_or(pred_c, target_c).sum()
        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(inter / union)
    return ious

def compute_pixel_accuracy(pred, target):
    """Compute overall pixel accuracy"""
    return (pred == target).sum() / pred.size

def compute_mean_accuracy(pred, target, n_classes):
    """Compute mean accuracy across classes"""
    accs = []
    for c in range(n_classes):
        mask = (target == c)
        if mask.sum() > 0:
            accs.append((pred[mask] == c).sum() / mask.sum())
    return np.mean(accs) if accs else 0

# ═══════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*50)
print("RUNNING VALIDATION")
print("="*50)

img_dir = os.path.join(VAL_DIR, 'Color_Images')
mask_dir = os.path.join(VAL_DIR, 'Segmentation')

if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
    print(f"⚠ Validation directories not found!")
    print(f"  Images: {img_dir}")
    print(f"  Masks: {mask_dir}")
    exit(1)

images = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
print(f"Found {len(images)} validation images\n")

# Store metrics
all_ious_per_class = []
all_pixel_accs = []
all_mean_accs = []

# Run validation
for fname in tqdm(images, desc="Validating"):
    # Load image and mask
    img = Image.open(os.path.join(img_dir, fname)).convert('RGB')
    mask = Image.open(os.path.join(mask_dir, fname))
    
    # Transform
    img_tensor = img_transform(img)
    mask_np = convert_mask_to_classes(mask)
    mask_resized = Image.fromarray(mask_np).resize((IMG_W, IMG_H), Image.NEAREST)
    mask_np_resized = np.array(mask_resized)
    
    # Predict
    pred = predict_merged(img_tensor)
    
    # Compute metrics
    ious = compute_iou_per_class(pred, mask_np_resized, N_CLASSES)
    all_ious_per_class.append(ious)
    
    pixel_acc = compute_pixel_accuracy(pred, mask_np_resized)
    all_pixel_accs.append(pixel_acc)
    
    mean_acc = compute_mean_accuracy(pred, mask_np_resized, N_CLASSES)
    all_mean_accs.append(mean_acc)

# ═══════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("VALIDATION RESULTS - MERGED MODEL (M2 + M3)")
print("="*60)

# Per-class IoU
ious_array = np.array(all_ious_per_class)
mean_ious_per_class = np.nanmean(ious_array, axis=0)

print("\n📊 Per-Class IoU:")
print("-" * 40)
for i, (name, miou) in enumerate(zip(CLASS_NAMES, mean_ious_per_class)):
    if not np.isnan(miou):
        bar = '█' * int(miou * 20)
        print(f"  {i:2d}. {name:<16}: {miou:.4f}  {bar}")
    else:
        print(f"  {i:2d}. {name:<16}: N/A (no pixels)")

# Overall metrics
mean_miou = np.nanmean(mean_ious_per_class)
mean_pixel_acc = np.mean(all_pixel_accs)
mean_mean_acc = np.mean(all_mean_accs)

print("\n" + "="*60)
print("📈 OVERALL METRICS:")
print("="*60)
print(f"  Mean IoU:           {mean_miou:.4f}  ({mean_miou*100:.2f}%)")
print(f"  Pixel Accuracy:     {mean_pixel_acc:.4f}  ({mean_pixel_acc*100:.2f}%)")
print(f"  Mean Accuracy:      {mean_mean_acc:.4f}  ({mean_mean_acc*100:.2f}%)")
print("="*60)

# Evaluation
print("\n🏆 EVALUATION:")
if mean_miou > 0.50:
    print("  🎉 EXCELLENT! mIoU > 0.50 - Great performance!")
elif mean_miou > 0.45:
    print("  ✓ GOOD! mIoU > 0.45 - Solid performance")
elif mean_miou > 0.40:
    print("  ~ FAIR mIoU > 0.40 - Acceptable")
else:
    print("  ⚠ LOW mIoU - Consider improvements")

print("\n" + "="*60)
