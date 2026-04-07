"""
PROPER ENSEMBLE - Soft Voting (Averaging Predictions, Not Weights)

This script:
1. Tests M1, M2, M3 individually
2. Creates proper ensemble by averaging predictions (soft voting)
3. Generates test predictions with proper mask values

M1: Fine-tuned model (best)
M2: Augmented model
M3: Hyperparameter tuned model
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from pathlib import Path
import zipfile

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Model paths
MEMBER1_PATH = 'submission/models/model_finetuned_best.pth.zip'  # M1
MEMBER2_PATH = 'model_augmented_best.pth'                        # M2
MEMBER3_PATH = 'model_member3_new.pth'                           # M3

# Data paths
VAL_DIR = 'Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/val'
TEST_DIR = 'Offroad_Segmentation_testImages/Offroad_Segmentation_testImages'

# Settings
IMG_H, IMG_W = 252, 462
TOKEN_H, TOKEN_W = 18, 33
N_CLASSES = 10
N_EMB = 384
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensemble weights for soft voting (prediction averaging)
ENSEMBLE_WEIGHTS = {
    'M1': 0.45,  # Best performer
    'M2': 0.30,
    'M3': 0.25
}

# Class mapping
VALUE_MAP = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

INV_VALUE_MAP = {v: k for k, v in VALUE_MAP.items()}

CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

print(f"Device: {DEVICE}")
print("="*60)
print("PROPER ENSEMBLE - Soft Voting")
print("="*60)

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
# LOAD BACKBONE
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("LOADING DINOv2 BACKBONE")
print("="*60)

backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', 
                          pretrained=True, verbose=False)
backbone = backbone.to(DEVICE).eval()
for p in backbone.parameters():
    p.requires_grad = False
print("✓ Backbone loaded")

# ═══════════════════════════════════════════════════════════════
# LOAD MODEL HEADS
# ═══════════════════════════════════════════════════════════════

def load_head(path, name):
    """Load a model head"""
    if not os.path.exists(path):
        print(f"⚠ {name}: Not found - {path}")
        return None
    
    resolved = path
    
    # Handle zip files
    if path.endswith('.zip'):
        extract_dir = path.replace('.zip', '_extracted')
        if not os.path.exists(extract_dir):
            print(f"  Extracting {path}...")
            with zipfile.ZipFile(path, 'r') as z:
                z.extractall(extract_dir)
        
        # Look for .pth
        pth_files = list(Path(extract_dir).rglob('*.pth'))
        if pth_files:
            resolved = str(pth_files[0])
            print(f"  Found .pth: {resolved}")
        else:
            # Directory format - try alternative approach
            model_dirs = [d for d in Path(extract_dir).iterdir() if d.is_dir() and (d / 'data').exists()]
            if model_dirs:
                # Load by creating model and loading checkpoint
                checkpoint_dir = str(model_dirs[0])
                print(f"  Loading directory model: {checkpoint_dir}")
                try:
                    # For directory format on Windows, we'll use the checkpoint file inside
                    # Actually, we need to load the pickle file
                    import pickle
                    data_pkl = os.path.join(checkpoint_dir, 'data.pkl')
                    if os.path.exists(data_pkl):
                        with open(data_pkl, 'rb') as f:
                            checkpoint = pickle.load(f)
                        if isinstance(checkpoint, dict):
                            if 'classifier_state' in checkpoint:
                                checkpoint = checkpoint['classifier_state']
                            elif 'state_dict' in checkpoint:
                                checkpoint = checkpoint['state_dict']
                        # Create head and load
                        head = SegmentationHeadConvNeXt(N_EMB, N_CLASSES, TOKEN_W, TOKEN_H).to(DEVICE)
                        head.load_state_dict(checkpoint, strict=True)
                        head.eval()
                        print(f"✓ {name}: Loaded from pickle")
                        return head
                except Exception as e:
                    print(f"  Directory load failed: {e}")
                    return None
            else:
                print(f"  No model found in zip")
                return None
    
    # Regular load
    try:
        checkpoint = torch.load(resolved, map_location=DEVICE, weights_only=False)
        
        # Handle formats
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
        
        head = SegmentationHeadConvNeXt(N_EMB, N_CLASSES, TOKEN_W, TOKEN_H).to(DEVICE)
        head.load_state_dict(state, strict=True)
        head.eval()
        print(f"✓ {name}: Loaded ({len(state)} tensors)")
        return head
    except Exception as e:
        print(f"⚠ {name}: Load failed - {e}")
        return None

print("\n" + "="*60)
print("LOADING MODEL HEADS")
print("="*60)

models = {}
for name, path in [('M1', MEMBER1_PATH), ('M2', MEMBER2_PATH), ('M3', MEMBER3_PATH)]:
    model = load_head(path, name)
    if model:
        models[name] = model

print(f"\n✓ Loaded {len(models)}/3 models")

if len(models) == 0:
    print("ERROR: No models loaded!")
    exit(1)

# Adjust weights for available models
total_weight = sum(ENSEMBLE_WEIGHTS[m] for m in models.keys())
ensemble_weights = {k: ENSEMBLE_WEIGHTS[k]/total_weight for k in models.keys()}

print(f"\nEnsemble weights (normalized):")
for name, w in ensemble_weights.items():
    print(f"  {name}: {w*100:.1f}%")

# ═══════════════════════════════════════════════════════════════
# TRANSFORMS & HELPERS
# ═══════════════════════════════════════════════════════════════

img_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def convert_mask_to_classes(mask_pil):
    """Convert mask to class IDs"""
    arr = np.array(mask_pil)
    out = np.zeros_like(arr, dtype=np.uint8)
    for raw, cls in VALUE_MAP.items():
        out[arr == raw] = cls
    return out

def pred_to_mask(pred):
    """Convert class indices to mask pixel values"""
    mask = np.zeros_like(pred, dtype=np.uint16)
    for cls, val in INV_VALUE_MAP.items():
        mask[pred == cls] = val
    return mask

# ═══════════════════════════════════════════════════════════════
# INFERENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_single_model(img_tensor, model_head):
    """Single model prediction with TTA"""
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    
    # Original
    feats = backbone.forward_features(img_tensor)['x_norm_patchtokens']
    logits = model_head(feats)
    logits = F.interpolate(logits, size=(IMG_H, IMG_W), mode='bilinear', align_corners=False)
    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    
    # TTA: Horizontal flip
    img_flip = torch.flip(img_tensor, dims=[-1])
    feats_flip = backbone.forward_features(img_flip)['x_norm_patchtokens']
    logits_flip = model_head(feats_flip)
    logits_flip = F.interpolate(logits_flip, size=(IMG_H, IMG_W), mode='bilinear', align_corners=False)
    probs_flip = F.softmax(logits_flip, dim=1).squeeze().cpu().numpy()
    probs_flip = np.flip(probs_flip, axis=-1)
    
    # Average TTA
    return (probs + probs_flip) / 2

@torch.no_grad()
def predict_ensemble(img_tensor):
    """Proper ensemble: average predictions (soft voting)"""
    img_tensor = img_tensor.to(DEVICE)
    
    # Average probabilities from all models
    combined_probs = np.zeros((N_CLASSES, IMG_H, IMG_W))
    
    for name, model_head in models.items():
        weight = ensemble_weights[name]
        probs = predict_single_model(img_tensor, model_head)
        combined_probs += weight * probs
    
    # Argmax to get final prediction
    pred = np.argmax(combined_probs, axis=0)
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
    return (pred == target).sum() / pred.size

def compute_mean_accuracy(pred, target, n_classes):
    accs = []
    for c in range(n_classes):
        mask = (target == c)
        if mask.sum() > 0:
            accs.append((pred[mask] == c).sum() / mask.sum())
    return np.mean(accs) if accs else 0

# ═══════════════════════════════════════════════════════════════
# OPTION 1: TEST INDIVIDUAL MODELS
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("OPTION 1: TESTING INDIVIDUAL MODELS")
print("="*60)

img_dir = os.path.join(VAL_DIR, 'Color_Images')
mask_dir = os.path.join(VAL_DIR, 'Segmentation')

if os.path.exists(img_dir) and os.path.exists(mask_dir):
    images = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    print(f"Found {len(images)} validation images")
    
    # Test each model individually
    individual_results = {}
    
    for mname, model_head in models.items():
        print(f"\nTesting {mname}...")
        all_ious = []
        
        for fname in tqdm(images[:100], desc=f"{mname}", leave=False):  # Test on 100 images
            img = Image.open(os.path.join(img_dir, fname)).convert('RGB')
            mask = Image.open(os.path.join(mask_dir, fname))
            
            img_tensor = img_transform(img)
            probs = predict_single_model(img_tensor, model_head)
            pred = np.argmax(probs, axis=0)
            
            mask_np = convert_mask_to_classes(mask)
            mask_resized = Image.fromarray(mask_np).resize((IMG_W, IMG_H), Image.NEAREST)
            mask_np_resized = np.array(mask_resized)
            
            ious = compute_iou_per_class(pred, mask_np_resized, N_CLASSES)
            all_ious.append(ious)
        
        # Calculate per-class and mean IoU
        ious_array = np.array(all_ious)
        mean_ious = np.nanmean(ious_array, axis=0)
        mean_miou = np.nanmean(mean_ious)
        
        individual_results[mname] = {
            'miou': mean_miou,
            'per_class': mean_ious
        }
        
        print(f"  {mname} Mean IoU: {mean_miou:.4f} ({mean_miou*100:.2f}%)")
    
    # Print comparison
    print("\n" + "="*60)
    print("INDIVIDUAL MODEL RESULTS (100 validation images):")
    print("="*60)
    for mname, results in individual_results.items():
        print(f"\n{mname}: mIoU = {results['miou']:.4f}")
        print("  Per-class IoU:")
        for i, (cls_name, iou) in enumerate(zip(CLASS_NAMES, results['per_class'])):
            if not np.isnan(iou):
                print(f"    {cls_name:<16}: {iou:.4f}")
    
    best_model = max(individual_results.items(), key=lambda x: x[1]['miou'])
    print(f"\n🏆 BEST MODEL: {best_model[0]} with mIoU = {best_model[1]['miou']:.4f}")

# ═══════════════════════════════════════════════════════════════
# OPTION 2: TEST ENSEMBLE
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("OPTION 2: TESTING ENSEMBLE (Soft Voting)")
print("="*60)

if os.path.exists(img_dir) and os.path.exists(mask_dir):
    print(f"Testing ensemble on {len(images)} images...")
    
    ensemble_ious = []
    ensemble_pixel_accs = []
    
    for fname in tqdm(images, desc="Ensemble"):
        img = Image.open(os.path.join(img_dir, fname)).convert('RGB')
        mask = Image.open(os.path.join(mask_dir, fname))
        
        img_tensor = img_transform(img)
        pred = predict_ensemble(img_tensor)
        
        mask_np = convert_mask_to_classes(mask)
        mask_resized = Image.fromarray(mask_np).resize((IMG_W, IMG_H), Image.NEAREST)
        mask_np_resized = np.array(mask_resized)
        
        ious = compute_iou_per_class(pred, mask_np_resized, N_CLASSES)
        ensemble_ious.append(ious)
        
        pixel_acc = compute_pixel_accuracy(pred, mask_np_resized)
        ensemble_pixel_accs.append(pixel_acc)
    
    # Calculate metrics
    ious_array = np.array(ensemble_ious)
    mean_ious = np.nanmean(ious_array, axis=0)
    mean_miou = np.nanmean(mean_ious)
    mean_pixel_acc = np.mean(ensemble_pixel_accs)
    
    print("\n" + "="*60)
    print("ENSEMBLE RESULTS (Soft Voting):")
    print("="*60)
    print(f"\n📊 Mean IoU:       {mean_miou:.4f} ({mean_miou*100:.2f}%)")
    print(f"📊 Pixel Accuracy:  {mean_pixel_acc:.4f} ({mean_pixel_acc*100:.2f}%)")
    
    print("\n📊 Per-Class IoU:")
    print("-" * 40)
    for i, (name, miou) in enumerate(zip(CLASS_NAMES, mean_ious)):
        if not np.isnan(miou):
            bar = '█' * int(miou * 20)
            print(f"  {i:2d}. {name:<16}: {miou:.4f}  {bar}")
    
    # Compare with best individual
    if individual_results:
        best_individual = max(r['miou'] for r in individual_results.values())
        improvement = mean_miou - best_individual
        print(f"\n📈 vs Best Individual: {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    # Evaluation
    print("\n🏆 EVALUATION:")
    if mean_miou > 0.50:
        print("  🎉 EXCELLENT! Ensemble mIoU > 0.50")
    elif mean_miou > 0.45:
        print("  ✓ GOOD! Ensemble mIoU > 0.45")
    elif mean_miou > 0.40:
        print("  ~ FAIR Ensemble mIoU > 0.40")
    else:
        print("  ⚠ LOW - Check model performance")

# ═══════════════════════════════════════════════════════════════
# GENERATE TEST PREDICTIONS
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("GENERATING TEST PREDICTIONS")
print("="*60)

test_img_dir = os.path.join(TEST_DIR, 'Color_Images')
output_dir = 'final_ensemble_predictions'

if not os.path.exists(test_img_dir):
    # Try alternative paths
    alt_paths = [
        'Offroad_Segmentation_testImages/Color_Images',
        'Offroad_Segmentation_testImages/Offroad_Segmentation_testImages/Color_Images',
    ]
    for alt in alt_paths:
        if os.path.exists(alt):
            test_img_dir = alt
            break

if os.path.exists(test_img_dir):
    os.makedirs(output_dir, exist_ok=True)
    test_images = sorted([f for f in os.listdir(test_img_dir) if f.endswith('.png')])
    
    print(f"Found {len(test_images)} test images")
    print(f"Generating predictions with PROPER ENSEMBLE...\n")
    
    for fname in tqdm(test_images, desc="Predicting"):
        img_path = os.path.join(test_img_dir, fname)
        img = Image.open(img_path).convert('RGB')
        img_tensor = img_transform(img)
        
        # Ensemble prediction with proper mask conversion
        pred = predict_ensemble(img_tensor)
        mask_pred = pred_to_mask(pred)
        
        base = Path(fname).stem
        Image.fromarray(mask_pred.astype(np.uint16)).save(f"{output_dir}/{base}_pred.png")
    
    print(f"\n✓ Saved {len(test_images)} predictions to {output_dir}/")
    
    # Verify
    sample_files = list(Path(output_dir).glob('*.png'))[:3]
    print(f"\nSample predictions:")
    for f in sample_files:
        img = Image.open(f)
        arr = np.array(img)
        unique = np.unique(arr)
        print(f"  {f.name}: unique values = {list(unique[:5])}...")
    
else:
    print(f"⚠ Test directory not found")

print("\n" + "="*60)
print("COMPLETE!")
print("="*60)
print(f"\n✓ Individual model testing done")
print(f"✓ Ensemble validation done")
print(f"✓ Test predictions saved to: {output_dir}/")
print("="*60)
