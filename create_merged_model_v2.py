"""
MERGED MODEL - Single Model Combining M2 and M3

Since M1 has loading issues on Windows, we'll create a merged model
from M2 (augmented) and M3 (hyperparameter tuned), which are the
most recent and best-performing models.

M2 weight: 50%
M3 weight: 50%
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

# Model paths (using M2 and M3 only due to M1 Windows compatibility issues)
MEMBER2_PATH = 'model_augmented_best.pth'      # Member 2 (50%)
MEMBER3_PATH = 'model_member3_new.pth'         # Member 3 (50%)

# Alternative: Try M1 path if available
MEMBER1_PATH = 'temp_member1_model'  # May fail on Windows - optional

# Data paths
TEST_DIR = 'Offroad_Segmentation_testImages/Offroad_Segmentation_testImages'

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

# Inverse mapping for saving
INV_VALUE_MAP = {v: k for k, v in VALUE_MAP.items()}

print(f"Device: {DEVICE}")
print(f"Creating MERGED MODEL from available members...")

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
# LOAD HELPER
# ═══════════════════════════════════════════════════════════════

def load_state_dict(path, name):
    """Load state dict from .pth file"""
    if not os.path.exists(path):
        print(f"⚠ {name}: Not found at {path}")
        return None
    
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'classifier_state' in checkpoint:
                state = checkpoint['classifier_state']
            elif 'state_dict' in checkpoint:
                state = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state = checkpoint['model']
            else:
                state = checkpoint
        elif hasattr(checkpoint, 'state_dict'):
            state = checkpoint.state_dict()
        else:
            state = checkpoint
        
        print(f"✓ {name}: Loaded {len(state)} tensors from {path}")
        return state
    except Exception as e:
        print(f"⚠ {name}: Load failed - {e}")
        return None

# ═══════════════════════════════════════════════════════════════
# CREATE MERGED MODEL
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*50)
print("LOADING MODEL WEIGHTS")
print("="*50)

states = {}
weights = {}

# Try to load M1 (optional - may fail on Windows with directory format)
m1_state = load_state_dict(MEMBER1_PATH, 'M1')
if m1_state:
    states['M1'] = m1_state
    weights['M1'] = 0.40
    print(f"  M1: 40% weight")

# Load M2 (should work)
m2_state = load_state_dict(MEMBER2_PATH, 'M2')
if m2_state:
    states['M2'] = m2_state
    weights['M2'] = 0.30 if m1_state else 0.50
    print(f"  M2: {weights['M2']*100:.0f}% weight")

# Load M3 (should work)
m3_state = load_state_dict(MEMBER3_PATH, 'M3')
if m3_state:
    states['M3'] = m3_state
    weights['M3'] = 0.30 if m1_state else 0.50
    print(f"  M3: {weights['M3']*100:.0f}% weight")

if len(states) < 2:
    print("ERROR: Need at least 2 models to merge!")
    exit(1)

print(f"\n✓ Loaded {len(states)} models for merging")

# Normalize weights to sum to 1
total_weight = sum(weights.values())
weights = {k: v/total_weight for k, v in weights.items()}

print(f"\nNormalized weights:")
for name, w in weights.items():
    print(f"  {name}: {w*100:.1f}%")

# Merge weights
print(f"\nMerging {len(states)} models...")
merged_state = {}
all_keys = set(states[list(states.keys())[0]].keys())

for key in all_keys:
    # Weighted average
    merged_param = sum(weights[name] * states[name][key] for name in states.keys())
    merged_state[key] = merged_param

print(f"✓ Merged {len(merged_state)} parameters")

# Create merged model
merged_head = SegmentationHeadConvNeXt(N_EMB, N_CLASSES, TOKEN_W, TOKEN_H).to(DEVICE)
merged_head.load_state_dict(merged_state, strict=True)
merged_head.eval()

print("✓ MERGED MODEL CREATED SUCCESSFULLY!")

# Save merged model
merged_path = 'model_merged_final.pth'
torch.save(merged_state, merged_path)
print(f"✓ Saved merged model: {merged_path} ({os.path.getsize(merged_path)/(1024*1024):.1f} MB)")

# ═══════════════════════════════════════════════════════════════
# LOAD BACKBONE
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*50)
print("LOADING DINOv2 BACKBONE")
print("="*50)

backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', 
                          pretrained=True, verbose=False)
backbone = backbone.to(DEVICE).eval()
for p in backbone.parameters():
    p.requires_grad = False

print("✓ Backbone loaded")

# ═══════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════

img_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def pred_to_mask(pred):
    """Convert class indices to mask pixel values (0, 100, 200, etc.)"""
    mask = np.zeros_like(pred, dtype=np.uint16)
    for cls, val in INV_VALUE_MAP.items():
        mask[pred == cls] = val
    return mask

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
# GENERATE PREDICTIONS
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*50)
print("GENERATING TEST PREDICTIONS")
print("="*50)

test_img_dir = os.path.join(TEST_DIR, 'Color_Images')
output_dir = 'merged_predictions'

if not os.path.exists(test_img_dir):
    print(f"⚠ Test directory not found: {test_img_dir}")
    print("Checking alternative paths...")
    alt_paths = [
        'Offroad_Segmentation_testImages/Color_Images',
        'Offroad_Segmentation_testImages/Offroad_Segmentation_testImages/Color_Images',
    ]
    for alt in alt_paths:
        if os.path.exists(alt):
            test_img_dir = alt
            print(f"✓ Found test images at: {alt}")
            break

if os.path.exists(test_img_dir):
    os.makedirs(output_dir, exist_ok=True)
    test_images = sorted([f for f in os.listdir(test_img_dir) if f.endswith('.png')])
    
    print(f"Found {len(test_images)} test images")
    print(f"Generating predictions with MERGED MODEL (TTA enabled)...\n")
    
    for fname in tqdm(test_images, desc="Predicting"):
        img_path = os.path.join(test_img_dir, fname)
        img = Image.open(img_path).convert('RGB')
        img_tensor = img_transform(img)
        
        # Predict with merged model
        pred = predict_merged(img_tensor)
        
        # Convert class indices to mask values and save
        mask_pred = pred_to_mask(pred)
        base = Path(fname).stem
        Image.fromarray(mask_pred.astype(np.uint16)).save(f"{output_dir}/{base}_pred.png")
    
    print(f"\n✓ Saved {len(test_images)} predictions to {output_dir}/")
    
    # Verify predictions are not all black
    sample_files = list(Path(output_dir).glob('*.png'))[:3]
    print(f"\nVerifying sample predictions:")
    for f in sample_files:
        img = Image.open(f)
        arr = np.array(img)
        unique = np.unique(arr)
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}: {size_kb:.1f} KB, unique values: {unique[:5]}...")
    
else:
    print(f"⚠ Test directory not found: {test_img_dir}")

print("\n" + "="*50)
print("COMPLETE!")
print("="*50)
print(f"\nMerged Model: {merged_path}")
print(f"Predictions: {output_dir}/")
print(f"\nThis is your SINGLE MODEL for final submission!")
print(f"Models merged: {', '.join(states.keys())}")
print("="*50)
