"""
Optimize Ensemble for Best mIoU
Tests different configurations to maximize performance
"""

import os
import sys
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
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

# Model paths
MEMBER1_PATH = 'submission/models/model_finetuned_best.pth.zip'
MEMBER2_PATH = 'submission/models/model_augmented_best.pth'
MEMBER3_PATH = 'submission/models/model_best.pth.zip'

# Data paths
VAL_IMAGES_DIR = 'desert-kaggle-api/val/Color_Images'
VAL_MASKS_DIR = 'desert-kaggle-api/val/Segmentation'

# Constants
IMG_H, IMG_W = 252, 462
TOKEN_H, TOKEN_W = 18, 33
N_CLASSES = 10
N_EMB = 384

VALUE_MAP = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

RARE_CLASSES = [5, 6]  # Ground Clutter, Logs

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
# HELPERS
# ═══════════════════════════════════════════════════════════════

def extract_zip(zip_path):
    if not zip_path.endswith('.zip'):
        return zip_path
    extract_dir = zip_path.replace('.zip', '_extracted')
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
    pth_files = list(Path(extract_dir).rglob('*.pth'))
    return str(pth_files[0]) if pth_files else zip_path

def load_head(ckpt_path, name):
    if not os.path.exists(ckpt_path):
        print(f"⚠ {name}: Not found")
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
        print(f"✓ {name}: Loaded")
        return head
    except Exception as e:
        print(f"⚠ {name}: Failed - {e}")
        return None

def convert_mask(mask_pil):
    arr = np.array(mask_pil)
    out = np.zeros_like(arr, dtype=np.uint8)
    for raw, cls in VALUE_MAP.items():
        out[arr == raw] = cls
    return out

def compute_iou(pred, target, n_classes=10):
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

# Transforms
img_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ═══════════════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("LOADING MODELS")
print("="*60)

# Load backbone
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', 
                          pretrained=True, verbose=False)
backbone = backbone.to(device).eval()
for p in backbone.parameters():
    p.requires_grad = False
print("✓ Backbone loaded\n")

# Load all heads
models = {}
m1 = load_head(MEMBER1_PATH, "Member 1 (Fine-tune)")
if m1:
    models['M1'] = m1
    
m2 = load_head(MEMBER2_PATH, "Member 2 (Augmented)")
if m2:
    models['M2'] = m2
    
m3 = load_head(MEMBER3_PATH, "Member 3 (Hyperparams)")
if m3:
    models['M3'] = m3

if not models:
    raise RuntimeError("No models loaded!")

print(f"\n✓ Loaded {len(models)} model(s): {list(models.keys())}")

# ═══════════════════════════════════════════════════════════════
# INFERENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_single(img_tensor, head):
    img_batch = img_tensor.unsqueeze(0).to(device)
    feats = backbone.forward_features(img_batch)['x_norm_patchtokens']
    logits = head(feats)
    logits = F.interpolate(logits, size=(IMG_H, IMG_W), mode='bilinear', align_corners=False)
    return F.softmax(logits, dim=1).squeeze(0)

@torch.no_grad()
def predict_ensemble_weighted(img_tensor, model_weights, use_tta=True):
    """Weighted ensemble prediction."""
    img_tensor = img_tensor.to(device)
    
    total_weight = sum(model_weights.values())
    combined = torch.zeros(N_CLASSES, IMG_H, IMG_W, device=device)
    
    for name, head in models.items():
        if name not in model_weights:
            continue
        weight = model_weights[name]
        
        # Original
        probs = predict_single(img_tensor.cpu(), head)
        
        if use_tta:
            # TTA: horizontal flip
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
# LOAD VALIDATION DATA
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("LOADING VALIDATION DATA")
print("="*60)

val_images = sorted([f for f in os.listdir(VAL_IMAGES_DIR) 
                     if f.lower().endswith('.png')])
print(f"Found {len(val_images)} validation images")

# Load all validation data
data = []
for fname in tqdm(val_images[:50], desc="Loading"):  # Use 50 for faster evaluation
    img_path = os.path.join(VAL_IMAGES_DIR, fname)
    mask_path = os.path.join(VAL_MASKS_DIR, fname)
    
    if not os.path.exists(mask_path):
        continue
    
    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path)
    
    img_tensor = img_transform(img)
    mask_np = convert_mask(mask)
    mask_resized = Image.fromarray(mask_np).resize((IMG_W, IMG_H), Image.NEAREST)
    mask_np_resized = np.array(mask_resized)
    
    data.append({
        'fname': fname,
        'img_tensor': img_tensor,
        'mask': mask_np_resized
    })

print(f"Loaded {len(data)} valid samples")

# ═══════════════════════════════════════════════════════════════
# EVALUATION FUNCTION
# ═══════════════════════════════════════════════════════════════

def evaluate_configuration(model_weights, use_tta, config_name):
    """Evaluate a specific configuration."""
    all_ious = []
    all_class_ious = []
    
    for sample in data:
        pred = predict_ensemble_weighted(sample['img_tensor'], model_weights, use_tta)
        miou, class_ious = compute_iou(pred, sample['mask'])
        all_ious.append(miou)
        all_class_ious.append(class_ious)
    
    mean_miou = np.nanmean(all_ious)
    mean_class_iou = np.nanmean(all_class_ious, axis=0)
    
    return {
        'name': config_name,
        'miou': mean_miou,
        'class_ious': mean_class_iou,
        'weights': model_weights.copy(),
        'tta': use_tta
    }

# ═══════════════════════════════════════════════════════════════
# TEST DIFFERENT CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TESTING CONFIGURATIONS")
print("="*60)

results = []

# 1. Individual models (no ensemble)
for name in models.keys():
    weights = {n: 1.0 if n == name else 0.0 for n in models.keys()}
    result = evaluate_configuration(weights, use_tta=True, config_name=f"{name} solo (TTA)")
    results.append(result)
    print(f"{result['name']}: mIoU = {result['miou']:.4f}")

# 2. Equal weight ensemble
if len(models) >= 2:
    equal_weight = 1.0 / len(models)
    weights = {n: equal_weight for n in models.keys()}
    result = evaluate_configuration(weights, use_tta=True, config_name="Equal weights (TTA)")
    results.append(result)
    print(f"{result['name']}: mIoU = {result['miou']:.4f}")

# 3. Different weight combinations (if all 3 models available)
if len(models) == 3:
    weight_combinations = [
        {'M1': 0.5, 'M2': 0.3, 'M3': 0.2},
        {'M1': 0.4, 'M2': 0.4, 'M3': 0.2},
        {'M1': 0.5, 'M2': 0.25, 'M3': 0.25},
        {'M1': 0.6, 'M2': 0.2, 'M3': 0.2},
        {'M1': 0.4, 'M2': 0.3, 'M3': 0.3},  # Original
        {'M1': 0.33, 'M2': 0.33, 'M3': 0.34},
        {'M1': 0.7, 'M2': 0.15, 'M3': 0.15},
    ]
    
    for w in weight_combinations:
        config_name = f"W{w['M1']:.1f}/{w['M2']:.1f}/{w['M3']:.1f}"
        result = evaluate_configuration(w, use_tta=True, config_name=config_name)
        results.append(result)
        print(f"{result['name']}: mIoU = {result['miou']:.4f}")

elif len(models) == 2:
    # Two model combinations
    model_names = list(models.keys())
    weight_combinations = [
        {model_names[0]: 0.6, model_names[1]: 0.4},
        {model_names[0]: 0.7, model_names[1]: 0.3},
        {model_names[0]: 0.5, model_names[1]: 0.5},
    ]
    
    for w in weight_combinations:
        config_name = f"{model_names[0]}/{model_names[1]} {w[model_names[0]]:.1f}/{w[model_names[1]]:.1f}"
        result = evaluate_configuration(w, use_tta=True, config_name=config_name)
        results.append(result)
        print(f"{result['name']}: mIoU = {result['miou']:.4f}")

# 4. Test without TTA for best configuration
if results:
    best = max(results, key=lambda x: x['miou'])
    result_no_tta = evaluate_configuration(best['weights'], use_tta=False, 
                                           config_name=best['name'] + " (no TTA)")
    results.append(result_no_tta)
    print(f"{result_no_tta['name']}: mIoU = {result_no_tta['miou']:.4f}")

# ═══════════════════════════════════════════════════════════════
# RESULTS SUMMARY
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("RESULTS RANKING")
print("="*60)

results_sorted = sorted(results, key=lambda x: x['miou'], reverse=True)

for i, r in enumerate(results_sorted[:10], 1):
    print(f"\n{i}. {r['name']}")
    print(f"   mIoU: {r['miou']:.4f}")
    print(f"   TTA: {'ON' if r['tta'] else 'OFF'}")
    if len(r['weights']) > 1:
        print(f"   Weights: {r['weights']}")

# ═══════════════════════════════════════════════════════════════
# TOP 2 CONFIGURATIONS DETAILS
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TOP 2 BEST CONFIGURATIONS")
print("="*60)

top2 = results_sorted[:2]

for i, r in enumerate(top2, 1):
    print(f"\n{'='*60}")
    print(f"CONFIGURATION {i}: {r['name']}")
    print(f"{'='*60}")
    print(f"mIoU: {r['miou']:.4f}")
    print(f"TTA: {'ON' if r['tta'] else 'OFF'}")
    print(f"Weights: {r['weights']}")
    print("\nPer-class IoU:")
    for cls_id, iou in enumerate(r['class_ious']):
        status = "✓" if not np.isnan(iou) and iou > 0.5 else "○" if not np.isnan(iou) else "✗"
        iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
        rare = " [RARE]" if cls_id in RARE_CLASSES else ""
        print(f"  {status} {CLASS_NAMES[cls_id]:<16}: {iou_str}{rare}")

# Save best configuration
print("\n" + "="*60)
print("RECOMMENDED CONFIGURATION FOR SUBMISSION")
print("="*60)
best = results_sorted[0]
print(f"\nUse: {best['name']}")
print(f"Expected mIoU: {best['miou']:.4f}")
print(f"\nCode to use:")
print(f"  weights = {best['weights']}")
print(f"  use_tta = {best['tta']}")

# Save to file
with open('best_config.txt', 'w') as f:
    f.write(f"Best Configuration: {best['name']}\n")
    f.write(f"mIoU: {best['miou']:.4f}\n")
    f.write(f"TTA: {best['tta']}\n")
    f.write(f"Weights: {best['weights']}\n\n")
    f.write("Per-class IoU:\n")
    for cls_id, iou in enumerate(best['class_ious']):
        iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
        f.write(f"  {CLASS_NAMES[cls_id]}: {iou_str}\n")

print(f"\n✓ Configuration saved to: best_config.txt")
