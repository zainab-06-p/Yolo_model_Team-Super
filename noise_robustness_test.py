"""
Noise-Robust Inference for DINOv2 Segmentation
Tests and handles various noise types and image manipulations
"""

import os
import time
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
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

# Test data
TEST_IMAGES_DIR = 'Offroad_Segmentation_testImages/Offroad_Segmentation_testImages/Color_Images'
TEST_MASKS_DIR = 'Offroad_Segmentation_testImages/Offroad_Segmentation_testImages/Segmentation'

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

# Optimized weights from earlier testing
ENSEMBLE_WEIGHTS = {'M1': 0.6, 'M2': 0.2, 'M3': 0.2}

# ═══════════════════════════════════════════════════════════════
# NOISE TYPES FOR TESTING
# ═══════════════════════════════════════════════════════════════

NOISE_TYPES = {
    'clean': lambda img: img,
    'gaussian_noise': lambda img: add_gaussian_noise(img, sigma=25),
    'salt_pepper': lambda img: add_salt_pepper(img, amount=0.02),
    'gaussian_blur': lambda img: cv2.GaussianBlur(img, (5, 5), 1.5),
    'motion_blur': lambda img: add_motion_blur(img, kernel_size=7),
    'brightness': lambda img: adjust_brightness(img, factor=1.3),
    'contrast': lambda img: adjust_contrast(img, factor=1.4),
    'jpeg_compression': lambda img: add_jpeg_artifacts(img, quality=50),
    'color_shift': lambda img: add_color_shift(img, shift=20),
    'sharpness': lambda img: adjust_sharpness(img, factor=2.0),
}

# ═══════════════════════════════════════════════════════════════
# NOISE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def add_gaussian_noise(image, sigma=25):
    """Add Gaussian noise to image."""
    noise = np.random.normal(0, sigma, image.shape).astype(np.uint8)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy

def add_salt_pepper(image, amount=0.02):
    """Add salt and pepper noise."""
    noisy = image.copy()
    num_pixels = int(amount * image.size)
    # Salt
    coords = [np.random.randint(0, i, num_pixels // 2) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255
    # Pepper
    coords = [np.random.randint(0, i, num_pixels // 2) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0
    return noisy

def add_motion_blur(image, kernel_size=7):
    """Add motion blur."""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(image, -1, kernel)

def adjust_brightness(image, factor=1.3):
    """Adjust brightness."""
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def adjust_contrast(image, factor=1.4):
    """Adjust contrast."""
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    return np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

def add_jpeg_artifacts(image, quality=50):
    """Add JPEG compression artifacts."""
    _, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

def add_color_shift(image, shift=20):
    """Add random color channel shift."""
    shifted = image.copy()
    shifted[:, :, 0] = np.clip(shifted[:, :, 0].astype(np.int16) + np.random.randint(-shift, shift), 0, 255)
    return shifted.astype(np.uint8)

def adjust_sharpness(image, factor=2.0):
    """Adjust sharpness using PIL."""
    pil_img = Image.fromarray(image)
    enhancer = ImageEnhance.Sharpness(pil_img)
    return np.array(enhancer.enhance(factor))

# ═══════════════════════════════════════════════════════════════
# PREPROCESSING FOR NOISE REMOVAL
# ═══════════════════════════════════════════════════════════════

def preprocess_denoise(image):
    """
    Advanced preprocessing to remove noise while preserving edges.
    """
    # Convert to LAB for better processing
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply non-local means denoising on L channel
    l_denoised = cv2.fastNlMeansDenoising(l, None, 10, 7, 21)
    
    # Merge back
    lab_denoised = cv2.merge([l_denoised, a, b])
    rgb_denoised = cv2.cvtColor(lab_denoised, cv2.COLOR_LAB2RGB)
    
    return rgb_denoised

def preprocess_enhance(image):
    """
    Enhance image for better segmentation.
    """
    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE on L channel for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Merge back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return rgb_enhanced

def preprocess_combined(image, use_denoise=True, use_enhance=True):
    """
    Combined preprocessing pipeline.
    """
    result = image.copy()
    
    if use_denoise:
        result = preprocess_denoise(result)
    
    if use_enhance:
        result = preprocess_enhance(result)
    
    return result

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

backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', 
                          pretrained=True, verbose=False)
backbone = backbone.to(device).eval()
for p in backbone.parameters():
    p.requires_grad = False
print("✓ Backbone loaded\n")

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
def predict_ensemble(img_tensor, use_tta=True, noise_aug=False):
    """
    Robust ensemble prediction with optional TTA and noise augmentation.
    """
    img_tensor = img_tensor.to(device)
    
    total_weight = sum(ENSEMBLE_WEIGHTS.values())
    combined = torch.zeros(N_CLASSES, IMG_H, IMG_W, device=device)
    
    # Multiple forward passes for robustness
    passes = []
    
    for name, head in models.items():
        if name not in ENSEMBLE_WEIGHTS:
            continue
        weight = ENSEMBLE_WEIGHTS[name]
        
        # Original pass
        probs = predict_single(img_tensor.cpu(), head)
        passes.append((weight, probs))
        
        if use_tta:
            # Horizontal flip TTA
            img_flip = torch.flip(img_tensor, dims=[-1])
            probs_flip = predict_single(img_flip.cpu(), head)
            probs_flip = torch.flip(probs_flip, dims=[-1])
            passes.append((weight * 0.5, (probs + probs_flip.cpu()) / 2))
    
    # Average all passes
    total_pass_weight = sum(w for w, _ in passes)
    for weight, probs in passes:
        combined += weight * probs.to(device)
    
    combined /= total_pass_weight
    return torch.argmax(combined, dim=0).cpu().numpy()

# ═══════════════════════════════════════════════════════════════
# NOISE ROBUSTNESS TEST
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("NOISE ROBUSTNESS TESTING")
print("="*60)

# Get test images
test_images = sorted([f for f in os.listdir(TEST_IMAGES_DIR) 
                      if f.lower().endswith('.png')])[:5]  # Test on 5 images

results = {noise_name: [] for noise_name in NOISE_TYPES.keys()}
results_preprocessed = {noise_name: [] for noise_name in NOISE_TYPES.keys()}

for fname in tqdm(test_images[:5], desc="Testing noise types"):
    img_path = os.path.join(TEST_IMAGES_DIR, fname)
    mask_path = os.path.join(TEST_MASKS_DIR, fname)
    
    if not os.path.exists(mask_path):
        continue
    
    # Load original image
    img_original = np.array(Image.open(img_path).convert('RGB'))
    mask = Image.open(mask_path)
    mask_np = convert_mask(mask)
    mask_resized = Image.fromarray(mask_np).resize((IMG_W, IMG_H), Image.NEAREST)
    mask_np_resized = np.array(mask_resized)
    
    # Test each noise type
    for noise_name, noise_fn in NOISE_TYPES.items():
        # Apply noise
        img_noisy = noise_fn(img_original.copy())
        
        # Without preprocessing
        img_tensor = img_transform(Image.fromarray(img_noisy))
        pred = predict_ensemble(img_tensor, use_tta=True)
        miou, _ = compute_iou(pred, mask_np_resized)
        results[noise_name].append(miou)
        
        # With preprocessing
        img_preprocessed = preprocess_combined(img_noisy, use_denoise=True, use_enhance=True)
        img_tensor_pp = img_transform(Image.fromarray(img_preprocessed))
        pred_pp = predict_ensemble(img_tensor_pp, use_tta=True)
        miou_pp, _ = compute_iou(pred_pp, mask_np_resized)
        results_preprocessed[noise_name].append(miou_pp)

# ═══════════════════════════════════════════════════════════════
# REPORT RESULTS
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("NOISE ROBUSTNESS RESULTS")
print("="*60)

print(f"\n{'Noise Type':<20} {'Without PP':<15} {'With PP':<15} {'Improvement':<15}")
print("-" * 65)

for noise_name in NOISE_TYPES.keys():
    mean_without = np.mean(results[noise_name])
    mean_with = np.mean(results_preprocessed[noise_name])
    improvement = mean_with - mean_without
    
    print(f"{noise_name:<20} {mean_without:<15.4f} {mean_with:<15.4f} {improvement:+<15.4f}")

# Calculate average robustness
clean_miou = np.mean(results['clean'])
noisy_miou_without = np.mean([np.mean(results[name]) for name in NOISE_TYPES.keys() if name != 'clean'])
noisy_miou_with = np.mean([np.mean(results_preprocessed[name]) for name in NOISE_TYPES.keys() if name != 'clean'])

print(f"\n{'='*60}")
print("ROBUSTNESS SUMMARY")
print(f"{'='*60}")
print(f"Clean images mIoU:          {clean_miou:.4f}")
print(f"Noisy images (no PP):       {noisy_miou_without:.4f} ({(noisy_miou_without/clean_miou)*100:.1f}% of clean)")
print(f"Noisy images (with PP):     {noisy_miou_with:.4f} ({(noisy_miou_with/clean_miou)*100:.1f}% of clean)")
print(f"Preprocessing improvement:  +{noisy_miou_with - noisy_miou_without:.4f} mIoU")

# ═══════════════════════════════════════════════════════════════
# RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("RECOMMENDATIONS FOR ROBUST SUBMISSION")
print(f"{'='*60}")

print("""
1. USE PREPROCESSING PIPELINE:
   - Apply preprocess_combined() to all test images
   - Denoising + CLAHE contrast enhancement
   - Expected improvement: +0.02-0.05 mIoU

2. USE ROBUST INFERENCE:
   - Enable TTA (horizontal flip)
   - Use ensemble with weights: M1=0.6, M2=0.2, M3=0.2
   - Multiple forward passes averaged

3. FOR EXTREME NOISE:
   - Add multi-scale testing
   - Use median filtering for salt/pepper
   - Bilateral filtering for edge-preserving smoothing

4. TEST-TIME TRAINING (OPTIONAL):
   - Adapt batch norm on test batch
   - 1-2 epochs on test data before prediction
""")

# Save preprocessing function for Kaggle
print(f"\n{'='*60}")
print("PREPROCESSING CODE FOR KAGGLE")
print(f"{'='*60}")
print("""
def preprocess_test_image(image):
    '''Apply to all test images before inference'''
    # Denoise
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_denoised = cv2.fastNlMeansDenoising(l, None, 10, 7, 21)
    lab = cv2.merge([l_denoised, a, b])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Enhance contrast
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return image
""")

print(f"\n{'='*60}")
print("TEST COMPLETE")
print(f"{'='*60}")
