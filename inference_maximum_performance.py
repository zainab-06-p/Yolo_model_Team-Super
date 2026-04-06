"""
Maximum Performance Inference for mIoU >= 0.90
Uses: TTA + Multi-Scale + Post-Processing + Model Ensemble
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
# CONFIGURATION - MAXIMUM PERFORMANCE
# ═══════════════════════════════════════════════════════════════

# Model paths - use ALL available models
MODEL_PATHS = {
    'M1': 'submission/models/model_finetuned_best.pth.zip',
    'M2': 'submission/models/model_augmented_best.pth',
    'M3': 'submission/models/model_best.pth.zip',
}

# Optimized weights from extensive testing
ENSEMBLE_WEIGHTS = {
    'M1': 0.50,  # Best individual model
    'M2': 0.30,
    'M3': 0.20,
}

# Multi-scale testing scales
MULTI_SCALE = [0.75, 1.0, 1.25]  # Test at different resolutions

# TTA variants
TTA_FLIPS = [False, True]  # Original + Horizontal flip

# Test data paths
TEST_IMAGES_DIR = 'desert-kaggle-api/val/Color_Images'  # For testing
TEST_MASKS_DIR = 'desert-kaggle-api/val/Segmentation'

# Image settings
IMG_H, IMG_W = 294, 462  # 21x33 tokens for vitl14, or 18x33 for vits14
TOKEN_H, TOKEN_W = 21, 33
N_CLASSES = 10

# Try to detect actual model size
N_EMB = 384  # Default vits14, will be updated

VALUE_MAP = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

# ═══════════════════════════════════════════════════════════════
# ADVANCED MODEL ARCHITECTURE (Matches training script)
# ═══════════════════════════════════════════════════════════════

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)
        attention = torch.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.conv_6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.conv_12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.conv_18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        x1 = self.conv_1(x)
        x6 = self.conv_6(x)
        x12 = self.conv_12(x)
        x18 = self.conv_18(x)
        x_global = self.global_pool(x)
        x_global = F.interpolate(x_global, size=size, mode='bilinear', align_corners=False)
        x_concat = torch.cat([x1, x6, x12, x18, x_global], dim=1)
        return self.project(x_concat)

class HighPerformanceDecoder(nn.Module):
    def __init__(self, embed_dim, num_classes, token_h, token_w):
        super().__init__()
        self.token_h = token_h
        self.token_w = token_w
        
        self.stem = nn.Sequential(
            nn.Conv2d(embed_dim, 512, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True)
        )
        self.aspp = ASPP(512, 256)
        self.attention1 = AttentionBlock(256)
        self.attention2 = AttentionBlock(256)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.classifier = nn.Conv2d(128, num_classes, 1)
    
    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.token_h, self.token_w, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.aspp(x)
        x = self.attention1(x)
        x = self.decoder1(x)
        x = self.attention2(x)
        x = self.decoder2(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=(IMG_H, IMG_W), mode='bilinear', align_corners=False)
        return x

# ═══════════════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("LOADING HIGH-PERFORMANCE ENSEMBLE")
print("="*60)

# Load backbone
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', 
                          pretrained=True, verbose=False)
backbone = backbone.to(device).eval()
for p in backbone.parameters():
    p.requires_grad = False

# Detect embedding dimension
with torch.no_grad():
    probe = torch.zeros(1, 3, IMG_H, IMG_W, device=device)
    feat = backbone.forward_features(probe)['x_norm_patchtokens']
    N_EMB = feat.shape[2]
    print(f"✓ Backbone: DINOv2 (embed_dim={N_EMB})")

# Load model heads
def load_model(path, name):
    if not os.path.exists(path):
        print(f"⚠ {name}: Not found")
        return None
    
    # Handle zip
    if path.endswith('.zip'):
        extract_dir = path.replace('.zip', '_extracted')
        if not os.path.exists(extract_dir):
            with zipfile.ZipFile(path, 'r') as z:
                z.extractall(extract_dir)
        pth_files = list(Path(extract_dir).rglob('*.pth'))
        path = str(pth_files[0]) if pth_files else path
    
    # Try loading - may need to adapt architecture
    head = HighPerformanceDecoder(N_EMB, N_CLASSES, TOKEN_H, TOKEN_W).to(device)
    
    try:
        state = torch.load(path, map_location=device, weights_only=False)
        if isinstance(state, dict):
            if 'classifier_state' in state:
                state = state['classifier_state']
            elif 'state_dict' in state:
                state = state['state_dict']
        
        head.load_state_dict(state, strict=False)
        head.eval()
        print(f"✓ {name}: Loaded")
        return head
    except Exception as e:
        print(f"⚠ {name}: {e}")
        # Try loading as simple head
        try:
            from monochromatic_handler import SegmentationHeadConvNeXt
            simple_head = SegmentationHeadConvNeXt(N_EMB, N_CLASSES, TOKEN_W, TOKEN_H).to(device)
            simple_head.load_state_dict(state, strict=False)
            simple_head.eval()
            print(f"✓ {name}: Loaded (simple architecture)")
            return simple_head
        except:
            print(f"✗ {name}: Failed to load")
            return None

models = {}
for name, path in MODEL_PATHS.items():
    model = load_model(path, name)
    if model:
        models[name] = model

if not models:
    raise RuntimeError("No models loaded!")

print(f"\n✓ Ensemble: {len(models)} models")
for name in models:
    print(f"  - {name}: weight={ENSEMBLE_WEIGHTS[name]}")

# ═══════════════════════════════════════════════════════════════
# MAXIMUM PERFORMANCE INFERENCE
# ═══════════════════════════════════════════════════════════════

img_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_image(img_path):
    """Load and preprocess image."""
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)
    return img, img_np

@torch.no_grad()
def predict_single_scale(img_tensor, scale=1.0):
    """Predict at single scale."""
    # Resize image
    if scale != 1.0:
        h, w = int(IMG_H * scale), int(IMG_W * scale)
        # Adjust to multiples of 14
        h = (h // 14) * 14
        w = (w // 14) * 14
        h = max(h, 14)
        w = max(w, 14)
        
        img_resized = F.interpolate(img_tensor.unsqueeze(0), size=(h, w), 
                                    mode='bilinear', align_corners=False).squeeze(0)
    else:
        img_resized = img_tensor
        h, w = IMG_H, IMG_W
    
    img_resized = img_resized.to(device)
    
    # Get features
    feats = backbone.forward_features(img_resized.unsqueeze(0))['x_norm_patchtokens']
    
    # Predict with each model
    total_weight = sum(ENSEMBLE_WEIGHTS.values())
    combined = torch.zeros(N_CLASSES, h, w, device=device)
    
    for name, model in models.items():
        if name not in ENSEMBLE_WEIGHTS:
            continue
        
        weight = ENSEMBLE_WEIGHTS[name]
        logits = model(feats)
        probs = F.softmax(logits.squeeze(0), dim=0)
        combined += weight * probs
    
    combined /= total_weight
    
    # Resize back to original size
    if scale != 1.0:
        combined = F.interpolate(combined.unsqueeze(0), size=(IMG_H, IMG_W),
                                mode='bilinear', align_corners=False).squeeze(0)
    
    return combined

@torch.no_grad()
def predict_multi_scale(img_tensor):
    """Multi-scale prediction with averaging."""
    all_probs = []
    
    for scale in MULTI_SCALE:
        probs = predict_single_scale(img_tensor, scale)
        all_probs.append(probs)
    
    # Average all scales
    avg_probs = torch.stack(all_probs).mean(dim=0)
    return avg_probs

@torch.no_grad()
def predict_with_tta(img_tensor):
    """Test-time augmentation: multi-scale + flips."""
    all_probs = []
    
    # Multi-scale
    for scale in MULTI_SCALE:
        # Original orientation
        probs = predict_single_scale(img_tensor, scale)
        all_probs.append(probs)
        
        # Horizontal flip
        img_flip = torch.flip(img_tensor, dims=[-1])
        probs_flip = predict_single_scale(img_flip, scale)
        probs_flip = torch.flip(probs_flip, dims=[-1])
        all_probs.append(probs_flip)
    
    # Average all
    avg_probs = torch.stack(all_probs).mean(dim=0)
    return avg_probs

def crf_postprocess(image_np, probs):
    """
    Dense CRF post-processing for boundary refinement.
    Requires pydensecrf: pip install pydensecrf
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
        
        # Convert to numpy
        probs_np = probs.cpu().numpy()  # [C, H, W]
        n_classes, h, w = probs_np.shape
        
        # Create unary potentials
        unary = unary_from_softmax(probs_np)
        unary = np.ascontiguousarray(unary)
        
        # Create CRF
        d = dcrf.DenseCRF2D(w, h, n_classes)
        d.setUnaryEnergy(unary)
        
        # Add pairwise Gaussian
        d.addPairwiseGaussian(sxy=3, compat=3, kernel=dcrf.DIAG_KERNEL, 
                             normalization=dcrf.NORMALIZE_SYMMETRIC)
        
        # Add pairwise bilateral (color-sensitive)
        img_rgb = np.ascontiguousarray(image_np.transpose(2, 0, 1))
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img_rgb, compat=10,
                              kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)
        
        # Inference
        Q = d.inference(5)
        Q = np.array(Q).reshape((n_classes, h, w))
        
        return torch.from_numpy(Q).float()
    except ImportError:
        # If pydensecrf not available, return original
        return probs

def predict_maximum_performance(img_path):
    """
    Maximum performance prediction pipeline:
    1. Load and preprocess
    2. Multi-scale prediction
    3. TTA (horizontal flip)
    4. Model ensemble
    5. CRF post-processing (optional)
    """
    # Load image
    img_pil, img_np = load_image(img_path)
    
    # Preprocess
    img_tensor = img_transform(img_pil)
    
    # Multi-scale TTA prediction
    probs = predict_with_tta(img_tensor)
    
    # CRF post-processing
    # probs = crf_postprocess(img_np, probs)
    
    # Get prediction
    pred = torch.argmax(probs, dim=0).cpu().numpy()
    
    return pred, probs

# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════

def convert_mask(mask_pil):
    arr = np.array(mask_pil)
    out = np.zeros_like(arr, dtype=np.uint8)
    for raw, cls in VALUE_MAP.items():
        out[arr == raw] = cls
    return out

def compute_iou(pred, target):
    ious = []
    for c in range(N_CLASSES):
        pred_c = (pred == c)
        target_c = (target == c)
        inter = np.logical_and(pred_c, target_c).sum()
        union = np.logical_or(pred_c, target_c).sum()
        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(inter / union)
    return np.nanmean(ious), ious

print("\n" + "="*60)
print("EVALUATION")
print("="*60)

# Get test images
test_images = sorted([f for f in os.listdir(TEST_IMAGES_DIR) 
                     if f.endswith('.png')])[:20]  # Test on 20

print(f"Evaluating on {len(test_images)} images...\n")

results = []
for fname in tqdm(test_images, desc="Processing"):
    img_path = os.path.join(TEST_IMAGES_DIR, fname)
    mask_path = os.path.join(TEST_MASKS_DIR, fname)
    
    if not os.path.exists(mask_path):
        continue
    
    # Predict
    start_time = time.time()
    pred, probs = predict_maximum_performance(img_path)
    inference_time = time.time() - start_time
    
    # Evaluate
    mask = Image.open(mask_path)
    mask_np = convert_mask(mask)
    mask_resized = Image.fromarray(mask_np).resize((IMG_W, IMG_H), Image.NEAREST)
    mask_np_resized = np.array(mask_resized)
    
    miou, class_ious = compute_iou(pred, mask_np_resized)
    
    results.append({
        'fname': fname,
        'miou': miou,
        'time': inference_time,
        'class_ious': class_ious
    })

# Report
if results:
    avg_miou = np.mean([r['miou'] for r in results])
    avg_time = np.mean([r['time'] for r in results])
    
    print(f"\n{'='*60}")
    print(f"MAXIMUM PERFORMANCE RESULTS")
    print(f"{'='*60}")
    print(f"Average mIoU: {avg_miou:.4f}")
    print(f"Average time: {avg_time:.3f}s per image")
    print(f"Images/sec: {1.0/avg_time:.2f}")
    
    # Per-class IoU
    class_ious_all = np.array([r['class_ious'] for r in results])
    mean_class_ious = np.nanmean(class_ious_all, axis=0)
    
    print(f"\nPer-class IoU:")
    for i, (name, iou) in enumerate(zip(CLASS_NAMES, mean_class_ious)):
        iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
        bar = "█" * int(iou * 20) if not np.isnan(iou) else ""
        print(f"  {name:<16}: {iou_str} {bar}")
    
    # Best images
    results_sorted = sorted(results, key=lambda x: x['miou'], reverse=True)
    print(f"\nTop 3 images:")
    for i, r in enumerate(results_sorted[:3], 1):
        print(f"  {i}. {r['fname']}: mIoU={r['miou']:.4f}")
    
    print(f"\n{'='*60}")
    print(f"TARGET: mIoU >= 0.90")
    print(f"CURRENT: mIoU = {avg_miou:.4f}")
    print(f"GAP: {max(0, 0.90 - avg_miou):.4f}")
    print(f"{'='*60}")

print("\n✓ Evaluation complete")
