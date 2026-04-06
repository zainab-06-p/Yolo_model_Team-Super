"""
Process 10 test images with ground truth and report metrics
"""

import os
import time
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

# Model paths
MEMBER1_PATH = 'submission/models/model_finetuned_best.pth.zip'
MEMBER2_PATH = 'submission/models/model_augmented_best.pth'
MEMBER3_PATH = 'submission/models/model_best.pth.zip'

# Test data paths
TEST_IMAGES_DIR = 'Offroad_Segmentation_testImages/Offroad_Segmentation_testImages/Color_Images'
TEST_MASKS_DIR = 'Offroad_Segmentation_testImages/Offroad_Segmentation_testImages/Segmentation'

# Process first 10 images
N_IMAGES = 10

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
print("LOADING ENSEMBLE MODELS")
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

# Use optimized weights from earlier testing
# Best: M1=0.6, M2=0.2, M3=0.2 with mIoU=0.4635
ENSEMBLE_WEIGHTS = {'M1': 0.6, 'M2': 0.2, 'M3': 0.2}
print(f"Using weights: {ENSEMBLE_WEIGHTS}")

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
def predict_ensemble(img_tensor, use_tta=True):
    """Ensemble prediction with TTA."""
    img_tensor = img_tensor.to(device)
    
    total_weight = sum(ENSEMBLE_WEIGHTS.values())
    combined = torch.zeros(N_CLASSES, IMG_H, IMG_W, device=device)
    
    for name, head in models.items():
        if name not in ENSEMBLE_WEIGHTS:
            continue
        weight = ENSEMBLE_WEIGHTS[name]
        
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
# PROCESS 10 TEST IMAGES
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("PROCESSING 10 TEST IMAGES")
print("="*60)

# Get first 10 images
test_images = sorted([f for f in os.listdir(TEST_IMAGES_DIR) 
                      if f.lower().endswith('.png')])[:N_IMAGES]

print(f"Processing {len(test_images)} images...\n")

results = []
total_time = 0

for i, fname in enumerate(test_images, 1):
    img_path = os.path.join(TEST_IMAGES_DIR, fname)
    mask_path = os.path.join(TEST_MASKS_DIR, fname)
    
    if not os.path.exists(mask_path):
        print(f"⚠ {fname}: Mask not found, skipping")
        continue
    
    # Load image and mask
    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path)
    
    img_tensor = img_transform(img)
    mask_np = convert_mask(mask)
    mask_resized = Image.fromarray(mask_np).resize((IMG_W, IMG_H), Image.NEAREST)
    mask_np_resized = np.array(mask_resized)
    
    # Time the inference
    start_time = time.time()
    pred = predict_ensemble(img_tensor, use_tta=True)
    inference_time = time.time() - start_time
    total_time += inference_time
    
    # Compute mIoU
    miou, class_ious = compute_iou(pred, mask_np_resized)
    
    results.append({
        'filename': fname,
        'miou': miou,
        'time': inference_time,
        'class_ious': class_ious
    })
    
    print(f"{i}. {fname}")
    print(f"   mIoU: {miou:.4f}")
    print(f"   Time: {inference_time:.3f}s")

# ═══════════════════════════════════════════════════════════════
# REPORT METRICS
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("FINAL REPORT")
print("="*60)

if results:
    # Average metrics
    avg_miou = np.mean([r['miou'] for r in results])
    avg_time = np.mean([r['time'] for r in results])
    total_time_sum = sum([r['time'] for r in results])
    
    print(f"\n📊 OVERALL METRICS ({len(results)} images)")
    print(f"   Average mIoU: {avg_miou:.4f}")
    print(f"   Average time per image: {avg_time:.3f}s")
    print(f"   Total processing time: {total_time_sum:.2f}s")
    print(f"   Images per second: {len(results)/total_time_sum:.2f}")
    
    # Per-class average IoU
    class_ious_all = np.array([r['class_ious'] for r in results])
    mean_class_ious = np.nanmean(class_ious_all, axis=0)
    
    print(f"\n📊 PER-CLASS AVERAGE IOU:")
    for cls_id, iou in enumerate(mean_class_ious):
        iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
        bar = "█" * int(iou * 20) if not np.isnan(iou) else ""
        print(f"   {CLASS_NAMES[cls_id]:<16}: {iou_str} {bar}")
    
    # Best and worst images
    best = max(results, key=lambda x: x['miou'])
    worst = min(results, key=lambda x: x['miou'])
    
    print(f"\n🏆 BEST IMAGE:")
    print(f"   {best['filename']}: mIoU = {best['miou']:.4f}")
    
    print(f"\n⚠ WORST IMAGE:")
    print(f"   {worst['filename']}: mIoU = {worst['miou']:.4f}")
    
    # Accuracy classification
    print(f"\n📊 ACCURACY CLASSIFICATION:")
    excellent = sum(1 for r in results if r['miou'] > 0.5)
    good = sum(1 for r in results if 0.4 < r['miou'] <= 0.5)
    fair = sum(1 for r in results if 0.3 < r['miou'] <= 0.4)
    poor = sum(1 for r in results if r['miou'] <= 0.3)
    
    print(f"   Excellent (>0.5): {excellent} images")
    print(f"   Good (0.4-0.5):   {good} images")
    print(f"   Fair (0.3-0.4):   {fair} images")
    print(f"   Poor (<0.3):      {poor} images")
    
    print(f"\n" + "="*60)
    print("SUMMARY FOR SUBMISSION")
    print("="*60)
    print(f"✓ Average mIoU: {avg_miou:.4f}")
    print(f"✓ Best mIoU:    {best['miou']:.4f}")
    print(f"✓ Processing:   {avg_time:.3f}s per image")
    print(f"✓ Total time:   {total_time_sum:.2f}s for {len(results)} images")
    print(f"\nModel is ready for submission!")
else:
    print("⚠ No results to report")
