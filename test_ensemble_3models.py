"""
3-Model Ensemble Test - Local Version
Tests M1, M2, and the NEW M3 together
"""

import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION - Update these paths
# ═══════════════════════════════════════════════════════════════

# Model paths (update to your local paths)
MEMBER1_PATH = 'submission/models/model_finetuned_best.pth.zip'  # Member 1 (fine-tuned) - will auto-extract
MEMBER2_PATH = 'model_augmented_best.pth'                        # Member 2 (augmented)  
MEMBER3_PATH = 'model_member3_new.pth'                           # NEW Member 3 (your trained model)

# Data paths
VAL_DIR = 'Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/val'
TEST_DIR = 'Offroad_Segmentation_testImages/Offroad_Segmentation_testImages'

# Settings
IMG_H, IMG_W = 252, 462
TOKEN_H, TOKEN_W = 18, 33
N_CLASSES = 10
N_EMB = 384
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensemble weights
ENSEMBLE_WEIGHTS = {
    'M1': 0.40,  # Best performer
    'M2': 0.30,
    'M3': 0.30   # New model
}

# Class mapping
VALUE_MAP = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

print(f"Device: {DEVICE}")
print(f"M1: {MEMBER1_PATH}")
print(f"M2: {MEMBER2_PATH}")
print(f"M3: {MEMBER3_PATH}")

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
# LOAD MODELS
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
print("✓ Backbone loaded\n")

# Load helpers
def load_head(ckpt_path, name):
    """Load a model head from checkpoint"""
    if not os.path.exists(ckpt_path):
        print(f"⚠ {name}: Not found at {ckpt_path}")
        return None
    
    # Handle zip files
    if ckpt_path.endswith('.zip'):
        import zipfile
        extract_dir = ckpt_path.replace('.zip', '_extracted')
        if not os.path.exists(extract_dir):
            print(f"  Extracting {ckpt_path}...")
            with zipfile.ZipFile(ckpt_path, 'r') as z:
                z.extractall(extract_dir)
        
        # Look for .pth files in extracted folder
        pth_files = list(Path(extract_dir).rglob('*.pth'))
        if pth_files:
            ckpt_path = str(pth_files[0])
            print(f"  Found .pth: {ckpt_path}")
        else:
            # Check if it's a PyTorch directory format (no .pth file)
            # Try to load from the directory directly
            print(f"  No .pth found, trying directory load...")
            ckpt_path = extract_dir
    
    head = SegmentationHeadConvNeXt(N_EMB, N_CLASSES, TOKEN_W, TOKEN_H).to(DEVICE)
    
    try:
        # Try loading as regular file
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(state, dict):
            if 'classifier_state' in state:
                state = state['classifier_state']
            elif 'state_dict' in state:
                state = state['state_dict']
            elif 'model' in state:
                state = state['model']
        
        head.load_state_dict(state, strict=True)
        head.eval()
        print(f"✓ {name}: Loaded successfully")
        return head
    except Exception as e:
        print(f"⚠ {name}: Load failed - {e}")
        return None

# Load all 3 models
models = {}
for name, path in [('M1', MEMBER1_PATH), ('M2', MEMBER2_PATH), ('M3', MEMBER3_PATH)]:
    model = load_head(path, f"Member {name}")
    if model:
        models[name] = model

print(f"\n{'='*50}")
if len(models) == 3:
    print("✓ ALL 3 MODELS LOADED SUCCESSFULLY!")
else:
    print(f"⚠ Only {len(models)}/3 models loaded")
    print("Update paths above and retry!")
print("="*50)

# ═══════════════════════════════════════════════════════════════
# INFERENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

img_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@torch.no_grad()
def predict_single(img_tensor, model):
    """Single model prediction"""
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    feats = backbone.forward_features(img_tensor)['x_norm_patchtokens']
    logits = model(feats)
    logits = F.interpolate(logits, size=(IMG_H, IMG_W), mode='bilinear', align_corners=False)
    return F.softmax(logits, dim=1).squeeze().cpu().numpy()

@torch.no_grad()
def predict_ensemble(img_tensor, use_tta=True):
    """3-model ensemble prediction with TTA"""
    if len(models) == 0:
        return None
    
    img_tensor = img_tensor.to(DEVICE)
    total_weight = sum(ENSEMBLE_WEIGHTS[name] for name in models.keys())
    combined = np.zeros((N_CLASSES, IMG_H, IMG_W))
    
    for name, head in models.items():
        weight = ENSEMBLE_WEIGHTS[name]
        
        # Original
        probs = predict_single(img_tensor, head)
        
        if use_tta:
            # Horizontal flip
            img_flip = torch.flip(img_tensor, dims=[-1])
            probs_flip = predict_single(img_flip, head)
            probs_flip = np.flip(probs_flip, axis=-1)
            probs = (probs + probs_flip) / 2
        
        combined += weight * probs
    
    combined /= total_weight
    return np.argmax(combined, axis=0)

# Inverse mapping for saving predictions
INV_VALUE_MAP = {v: k for k, v in VALUE_MAP.items()}

def pred_to_mask(pred):
    """Convert class indices to mask pixel values"""
    mask = np.zeros_like(pred, dtype=np.uint16)
    for cls, val in INV_VALUE_MAP.items():
        mask[pred == cls] = val
    return mask

def convert_mask(mask_pil):
    """Convert mask to class IDs"""
    arr = np.array(mask_pil)
    out = np.zeros_like(arr, dtype=np.uint8)
    for raw, cls in VALUE_MAP.items():
        out[arr == raw] = cls
    return out

def compute_iou(pred, target):
    """Compute mean IoU"""
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
    return np.nanmean(ious)

# ═══════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*50)
print("VALIDATION")
print("="*50)

if os.path.exists(VAL_DIR):
    img_dir = os.path.join(VAL_DIR, 'Color_Images')
    mask_dir = os.path.join(VAL_DIR, 'Segmentation')
    
    if os.path.exists(img_dir) and os.path.exists(mask_dir):
        images = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        
        print(f"Found {len(images)} validation images")
        print("Testing on 50 images...\n")
        
        # Test individual models
        individual_results = {}
        for mname, model in models.items():
            ious = []
            for fname in tqdm(images[:50], desc=f"Testing {mname}", leave=False):
                img = Image.open(os.path.join(img_dir, fname)).convert('RGB')
                mask = Image.open(os.path.join(mask_dir, fname))
                
                img_tensor = img_transform(img)
                pred = predict_single(img_tensor, model).argmax(axis=0)
                
                mask_np = convert_mask(mask)
                mask_resized = Image.fromarray(mask_np).resize((IMG_W, IMG_H), Image.NEAREST)
                mask_np_resized = np.array(mask_resized)
                
                miou = compute_iou(pred, mask_np_resized)
                ious.append(miou)
            
            mean_miou = np.nanmean(ious)
            individual_results[mname] = mean_miou
            print(f"{mname} mIoU: {mean_miou:.4f}")
        
        # Test ensemble
        print("\nTesting 3-Model Ensemble...")
        ensemble_ious = []
        for fname in tqdm(images[:50], desc="Ensemble"):
            img = Image.open(os.path.join(img_dir, fname)).convert('RGB')
            mask = Image.open(os.path.join(mask_dir, fname))
            
            img_tensor = img_transform(img)
            pred = predict_ensemble(img_tensor, use_tta=True)
            
            mask_np = convert_mask(mask)
            mask_resized = Image.fromarray(mask_np).resize((IMG_W, IMG_H), Image.NEAREST)
            mask_np_resized = np.array(mask_resized)
            
            miou = compute_iou(pred, mask_np_resized)
            ensemble_ious.append(miou)
        
        ensemble_miou = np.nanmean(ensemble_ious)
        
        print("\n" + "="*50)
        print("FINAL RESULTS")
        print("="*50)
        for mname, miou in individual_results.items():
            print(f"{mname}: {miou:.4f} mIoU")
        print(f"\n3-Model Ensemble: {ensemble_miou:.4f} mIoU")
        print(f"Improvement: +{ensemble_miou - max(individual_results.values()):.4f}")
        print("="*50)
        
        if ensemble_miou > 0.50:
            print("🎉 EXCELLENT! Ensemble exceeds 0.50 mIoU target!")
        elif ensemble_miou > 0.45:
            print("✓ GOOD! Ensemble performing well!")
        else:
            print("⚠ Consider retraining or adjusting weights")
            
    else:
        print(f"⚠ Validation directories not found")
else:
    print(f"⚠ Validation directory not found: {VAL_DIR}")

# ═══════════════════════════════════════════════════════════════
# TEST PREDICTIONS
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*50)
print("TEST PREDICTIONS")
print("="*50)

test_img_dir = os.path.join(TEST_DIR, 'Color_Images')
output_dir = 'ensemble_predictions'

if os.path.exists(test_img_dir):
    os.makedirs(output_dir, exist_ok=True)
    test_images = sorted([f for f in os.listdir(test_img_dir) if f.endswith('.png')])
    
    print(f"Generating predictions for {len(test_images)} test images...")
    
    for fname in tqdm(test_images, desc="Predicting"):
        img_path = os.path.join(test_img_dir, fname)
        img = Image.open(img_path).convert('RGB')
        img_tensor = img_transform(img)
        
        pred = predict_ensemble(img_tensor, use_tta=True)
        
        # Save prediction with proper mask values
        base = Path(fname).stem
        mask_pred = pred_to_mask(pred)
        Image.fromarray(mask_pred.astype(np.uint16)).save(f"{output_dir}/{base}_pred.png")
    
    print(f"✓ Saved {len(test_images)} predictions to {output_dir}/")
else:
    print(f"⚠ Test directory not found: {test_img_dir}")

print("\n" + "="*50)
print("COMPLETE!")
print("="*50)
