"""
Ensemble Inference for Kaggle
DINOv2 Offroad Segmentation - 3 Model Ensemble
"""

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION - UPDATE THESE PATHS
# ═══════════════════════════════════════════════════════════════

# MANUAL PATHS - Set these if you know the exact paths
# Leave as None to use auto-detection
MANUAL_MEMBER1_PATH = None  # e.g., '/kaggle/input/models/model_finetuned_best.pth.zip'
MANUAL_MEMBER2_PATH = None  # e.g., '/kaggle/input/models/model_augmented_best.pth'
MANUAL_MEMBER3_PATH = None  # e.g., '/kaggle/input/models/model_best.pth.zip'

# Dataset path (your main dataset)
DATASET_PATH = '/kaggle/input/yolo-training-data'

# ═══════════════════════════════════════════════════════════════
# AUTO-DETECTION FUNCTION
# ═══════════════════════════════════════════════════════════════

def find_model_file(base_path, expected_name):
    """
    Search for model file in common locations.
    Handles both files and directories containing .pth or .zip files.
    """
    import glob
    
    if base_path is None:
        return None
    
    # If it's a file, return it directly
    if os.path.isfile(base_path):
        return base_path
    
    # If it's a directory, search for .pth or .zip files inside
    if os.path.isdir(base_path):
        # Look for .pth files
        pth_files = glob.glob(os.path.join(base_path, '*.pth'))
        if pth_files:
            return pth_files[0]
        
        # Look for .zip files
        zip_files = glob.glob(os.path.join(base_path, '*.zip'))
        if zip_files:
            return zip_files[0]
        
        # Recursive search
        for root, dirs, files in os.walk(base_path):
            for f in files:
                if f.endswith('.pth') or f.endswith('.zip'):
                    return os.path.join(root, f)
    
    return None

# Auto-detect if manual paths not set
if MANUAL_MEMBER1_PATH:
    MEMBER1_PATH = MANUAL_MEMBER1_PATH
else:
    # Try common locations
    MEMBER1_PATH = find_model_file('/kaggle/working/model_finetuned_best.pth', 'model_finetuned')
    if not MEMBER1_PATH:
        MEMBER1_PATH = find_model_file('/kaggle/input/yolo-training-data-adi/model_finetuned_best.pth', 'model_finetuned')
    if not MEMBER1_PATH:
        MEMBER1_PATH = find_model_file('/kaggle/input/model-finetuned-best', 'model_finetuned')

if MANUAL_MEMBER2_PATH:
    MEMBER2_PATH = MANUAL_MEMBER2_PATH
else:
    MEMBER2_PATH = find_model_file('/kaggle/working/model_augmented_best.pth', 'model_augmented')
    if not MEMBER2_PATH:
        MEMBER2_PATH = find_model_file('/kaggle/input/yolo-training-data-adi/model_augmented_best.pth', 'model_augmented')
    if not MEMBER2_PATH:
        MEMBER2_PATH = find_model_file('/kaggle/input/model-augmented-best', 'model_augmented')

if MANUAL_MEMBER3_PATH:
    MEMBER3_PATH = MANUAL_MEMBER3_PATH
else:
    MEMBER3_PATH = find_model_file('/kaggle/working/model_best.pth', 'model_best')
    if not MEMBER3_PATH:
        MEMBER3_PATH = find_model_file('/kaggle/input/yolo-training-data-adi/model_best.pth', 'model_best')
    if not MEMBER3_PATH:
        MEMBER3_PATH = find_model_file('/kaggle/input/model-best', 'model_best')

# ═══════════════════════════════════════════════════════════════
# PRINT DETECTED PATHS
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*50)
print("MODEL PATH DETECTION")
print("="*50)
print(f"Member 1: {MEMBER1_PATH if MEMBER1_PATH else 'NOT FOUND'}")
print(f"Member 2: {MEMBER2_PATH if MEMBER2_PATH else 'NOT FOUND'}")
print(f"Member 3: {MEMBER3_PATH if MEMBER3_PATH else 'NOT FOUND'}")
print("="*50)
# ═══════════════════════════════════════════════════════════════

import os
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ═══════════════════════════════════════════════════════════════
# ROBUST PREPROCESSING FOR NOISE/MANIPULATED IMAGES
# ═══════════════════════════════════════════════════════════════

USE_ROBUST_PREPROCESSING = True  # Enable for noisy/manipulated test images

def robust_preprocess(image_np):
    """
    Advanced preprocessing for noise removal and image enhancement.
    Handles: Gaussian noise, blur, compression artifacts, brightness/contrast issues
    """
    if not USE_ROBUST_PREPROCESSING:
        return image_np
    
    # Convert to LAB color space for better processing
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Step 1: Denoise L channel (luminance) - preserves edges
    l_denoised = cv2.fastNlMeansDenoising(l, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Step 2: CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_denoised)
    
    # Merge back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return result

def detect_noise_level(image_np):
    """
    Detect if image has significant noise/blur issues.
    Returns: 'clean', 'noisy', or 'blurred'
    """
    # Convert to grayscale
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
    
    # Calculate Laplacian variance (edge sharpness)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Estimate noise using median absolute deviation
    median = np.median(gray)
    mad = np.median(np.abs(gray - median))
    noise_estimate = mad / 0.6745
    
    # Classify
    if lap_var < 100:
        return 'blurred'
    elif noise_estimate > 30:
        return 'noisy'
    else:
        return 'clean'

print(f"Robust preprocessing: {'ENABLED' if USE_ROBUST_PREPROCESSING else 'DISABLED'}")

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

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

COLOR_PALETTE = np.array([
    [0, 0, 0], [34, 139, 34], [0, 255, 0], [210, 180, 140],
    [139, 90, 43], [128, 128, 0], [139, 69, 19], [128, 128, 128],
    [160, 82, 45], [135, 206, 235]
], dtype=np.uint8)

# ═══════════════════════════════════════════════════════════════
# SPECIALIZED HANDLING FOR MONOCHROMATIC BLACK IMAGES
# ═══════════════════════════════════════════════════════════════

ENABLE_BLACK_IMAGE_HANDLING = True

def analyze_black_content(image_np):
    """Analyze how much of the image is black/monochromatic."""
    if len(image_np.shape) == 3:
        black_mask = np.all(image_np < 20, axis=2)
        dark_mask = np.all(image_np < 50, axis=2)
        black_percent = np.sum(black_mask) / black_mask.size * 100
        dark_percent = np.sum(dark_mask) / dark_mask.size * 100
        channel_diff = np.std(image_np, axis=2)
        mono_mask = channel_diff < 10
        mono_percent = np.sum(mono_mask) / mono_mask.size * 100
        return {
            'black_percent': black_percent,
            'dark_percent': dark_percent,
            'mono_percent': mono_percent,
            'is_monochromatic': mono_percent > 30,
            'is_dark': dark_percent > 40
        }
    return {'is_monochromatic': False, 'is_dark': False}

def enhance_monochromatic_black(image_np):
    """
    Specialized enhancement for monochromatic black images with black sky.
    Uses texture analysis and local contrast to separate sky from objects.
    """
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
    
    # Step 1: Texture analysis to distinguish sky (smooth) from objects (textured)
    local_mean = cv2.blur(gray.astype(np.float32), (5, 5))
    local_mean_sq = cv2.blur((gray.astype(np.float32) ** 2), (5, 5))
    local_std = np.sqrt(np.maximum(local_mean_sq - local_mean ** 2, 0) + 1e-8)
    local_std_norm = (local_std / (local_std.max() + 1e-8) * 255).astype(np.uint8)
    
    # Sky is smooth (low std), objects have texture (high std)
    sky_mask = local_std_norm < 15
    object_mask = local_std_norm >= 15
    
    # Step 2: Enhance in LAB color space
    if len(image_np.shape) == 3:
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Enhance luminance with high contrast CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Boost color channels to create artificial distinction
        a_boosted = cv2.addWeighted(a, 2.0, a, 0, 0)
        b_boosted = cv2.addWeighted(b, 2.0, b, 0, 0)
        
        # For sky: add slight blue tint to distinguish from background
        sky_pixels = sky_mask
        b_boosted[sky_pixels] = np.clip(b_boosted[sky_pixels] + 30, 0, 255)
        
        # For objects: enhance local contrast
        l_enhanced[object_mask] = cv2.addWeighted(l[object_mask], 1.8, 
                                                   l_enhanced[object_mask], 0, 20)
        
        lab_enhanced = cv2.merge([l_enhanced, a_boosted, b_boosted])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        # Step 3: Edge enhancement for object boundaries
        edges = cv2.Canny(gray, 30, 100)
        kernel = np.ones((2, 2), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Sharpen edges
        result[edges_dilated > 0] = np.clip(
            result[edges_dilated > 0].astype(np.int16) + 40, 0, 255
        ).astype(np.uint8)
    else:
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        result = clahe.apply(gray)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    
    return result

def handle_monochromatic_black(image_np):
    """Complete pipeline for handling monochromatic black images."""
    if not ENABLE_BLACK_IMAGE_HANDLING:
        return image_np, False
    
    analysis = analyze_black_content(image_np)
    
    if analysis['is_monochromatic'] or analysis['dark_percent'] > 30:
        enhanced = enhance_monochromatic_black(image_np)
        return enhanced, True
    
    return image_np, False

print(f"Black image handling: {'ENABLED' if ENABLE_BLACK_IMAGE_HANDLING else 'DISABLED'}")

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
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
    pth_files = list(Path(extract_dir).rglob('*.pth'))
    return str(pth_files[0]) if pth_files else zip_path

def load_head(ckpt_path, name):
    """Load model head from checkpoint."""
    if ckpt_path is None:
        print(f"⚠ {name}: Path is None (not found)")
        return None
    
    if not os.path.exists(ckpt_path):
        print(f"⚠ {name}: Path does not exist: {ckpt_path}")
        return None
    
    # If it's a directory, try to find the actual file
    if os.path.isdir(ckpt_path):
        print(f"⚠ {name}: Path is a directory, searching inside: {ckpt_path}")
        import glob
        pth_files = glob.glob(os.path.join(ckpt_path, '*.pth')) + glob.glob(os.path.join(ckpt_path, '*.zip'))
        if pth_files:
            ckpt_path = pth_files[0]
            print(f"  Found: {ckpt_path}")
        else:
            print(f"  No .pth or .zip files found in directory")
            return None
    
    # Extract if zip
    resolved = extract_zip(ckpt_path)
    
    head = SegmentationHeadConvNeXt(N_EMB, N_CLASSES, TOKEN_W, TOKEN_H).to(device)
    
    try:
        state = torch.load(resolved, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
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
        print(f"✓ {name}: Loaded successfully")
        return head
    except Exception as e:
        print(f"⚠ {name}: Load failed - {e}")
        return None

# ═══════════════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*50)
print("LOADING MODELS")
print("="*50)

# Load backbone
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', 
                          pretrained=True, verbose=False)
backbone = backbone.to(device).eval()
for p in backbone.parameters():
    p.requires_grad = False
print("✓ DINOv2 backbone loaded")

# Load heads
heads = []
weights = []

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
    raise RuntimeError("No models loaded! Check MEMBER*_PATH variables.")

print(f"\n✓ Loaded {len(heads)} model(s)")
for name, _, w in heads:
    print(f"  - {name}: weight={w}")

# ═══════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════

def convert_mask(mask_pil):
    arr = np.array(mask_pil)
    out = np.zeros_like(arr, dtype=np.uint8)
    for raw, cls in VALUE_MAP.items():
        out[arr == raw] = cls
    return out

img_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

mask_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

class ValDataset(Dataset):
    def __init__(self, data_dir):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.mask_dir = os.path.join(data_dir, 'Segmentation')
        self.filenames = sorted(os.listdir(self.image_dir)) if os.path.exists(self.image_dir) else []
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = img_transform(Image.open(os.path.join(self.image_dir, fname)).convert('RGB'))
        mask = Image.open(os.path.join(self.mask_dir, fname))
        mask_cls = Image.fromarray(convert_mask(mask))
        mask_t = (mask_transform(mask_cls) * 255).long().squeeze(0)
        return img, mask_t, fname

class TestDataset(Dataset):
    def __init__(self, data_dir):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.filenames = sorted(os.listdir(self.image_dir)) if os.path.exists(self.image_dir) else []
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.image_dir, fname)
        
        # Load image as numpy for preprocessing
        img_pil = Image.open(img_path).convert('RGB')
        img_np = np.array(img_pil)
        
        # Step 1: Handle monochromatic black images (black sky, dark objects)
        img_np, was_black_enhanced = handle_monochromatic_black(img_np)
        if was_black_enhanced:
            print(f"  [Black-enhanced: {fname}]")
        
        # Step 2: Apply robust preprocessing for noise/blur if enabled
        if USE_ROBUST_PREPROCESSING and not was_black_enhanced:
            noise_type = detect_noise_level(img_np)
            if noise_type in ['noisy', 'blurred']:
                img_np = robust_preprocess(img_np)
                print(f"  [Preprocessed {fname}: {noise_type}]")
        
        # Convert back to PIL and apply transforms
        img_tensor = img_transform(Image.fromarray(img_np))
        return img_tensor, fname

# ═══════════════════════════════════════════════════════════════
# SETUP DATA
# ═══════════════════════════════════════════════════════════════

val_dir = os.path.join(DATASET_PATH, 'Offroad_Segmentation_Training_Dataset', 'val')
test_dir = os.path.join(DATASET_PATH, 'Offroad_Segmentation_testImages')

print(f"\nVal dir: {val_dir}")
print(f"Test dir: {test_dir}")

# Check if paths exist
if not os.path.exists(val_dir):
    print(f"⚠ Val dir not found: {val_dir}")
    # Try alternate structure
    val_dir = os.path.join(DATASET_PATH, 'val')
    if os.path.exists(val_dir):
        print(f"  Found alternate: {val_dir}")
        
if not os.path.exists(test_dir):
    print(f"⚠ Test dir not found: {test_dir}")
    test_dir = os.path.join(DATASET_PATH, 'test')
    if os.path.exists(test_dir):
        print(f"  Found alternate: {test_dir}")

val_dataset = ValDataset(val_dir) if os.path.exists(val_dir) else None
test_dataset = TestDataset(test_dir) if os.path.exists(test_dir) else None

print(f"\nVal images: {len(val_dataset) if val_dataset else 'N/A'}")
print(f"Test images: {len(test_dataset) if test_dataset else 'N/A'}")

# ═══════════════════════════════════════════════════════════════
# INFERENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_single(img, head):
    """Single model prediction."""
    img = img.to(device).unsqueeze(0)
    feats = backbone.forward_features(img)['x_norm_patchtokens']
    logits = head(feats)
    logits = F.interpolate(logits, size=(IMG_H, IMG_W), mode='bilinear', align_corners=False)
    return F.softmax(logits, dim=1).squeeze(0)

@torch.no_grad()
def predict_ensemble(img, use_tta=True):
    """Ensemble prediction with optional TTA."""
    img = img.to(device)
    
    total_weight = sum(w for _, _, w in heads)
    combined = torch.zeros(N_CLASSES, IMG_H, IMG_W, device=device)
    
    for name, head, weight in heads:
        # Original
        probs = predict_single(img, head)
        
        if use_tta:
            # Horizontal flip TTA
            img_flip = torch.flip(img, dims=[-1])
            probs_flip = predict_single(img_flip, head)
            probs_flip = torch.flip(probs_flip, dims=[-1])
            probs = (probs + probs_flip) / 2
        
        combined += weight * probs
    
    combined /= total_weight
    return torch.argmax(combined, dim=0).cpu().numpy()

def compute_iou(pred, target):
    """Compute mean IoU."""
    pred = pred.flatten()
    target = target.flatten()
    ious = []
    for c in range(N_CLASSES):
        pc = pred == c
        tc = target == c
        inter = np.logical_and(pc, tc).sum()
        union = np.logical_or(pc, tc).sum()
        ious.append(inter / union if union > 0 else np.nan)
    return np.nanmean(ious), ious

# ═══════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════

if val_dataset and len(val_dataset) > 0:
    print("\n" + "="*50)
    print("VALIDATION")
    print("="*50)
    
    all_ious = []
    all_class_ious = []
    
    for i in tqdm(range(len(val_dataset)), desc="Val"):
        img, mask, _ = val_dataset[i]
        pred = predict_ensemble(img, use_tta=True)
        miou, class_ious = compute_iou(pred, mask.numpy())
        all_ious.append(miou)
        all_class_ious.append(class_ious)
    
    mean_iou = np.nanmean(all_ious)
    mean_class_iou = np.nanmean(all_class_ious, axis=0)
    
    print(f"\nMean IoU: {mean_iou:.4f}")
    print("\nPer-class IoU:")
    for i, (name, iou) in enumerate(zip(CLASS_NAMES, mean_class_iou)):
        print(f"  {name}: {iou:.4f}")

# ═══════════════════════════════════════════════════════════════
# TEST PREDICTIONS
# ═══════════════════════════════════════════════════════════════

if test_dataset and len(test_dataset) > 0:
    print("\n" + "="*50)
    print("TEST PREDICTIONS")
    print("="*50)
    
    output_dir = '/kaggle/working/predictions'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/masks", exist_ok=True)
    os.makedirs(f"{output_dir}/color", exist_ok=True)
    
    for i in tqdm(range(len(test_dataset)), desc="Test"):
        img, fname = test_dataset[i]
        pred = predict_ensemble(img, use_tta=True)
        
        # Save raw mask
        base = Path(fname).stem
        Image.fromarray(pred.astype(np.uint8)).save(f"{output_dir}/masks/{base}_pred.png")
        
        # Save colorized
        color = COLOR_PALETTE[pred]
        import cv2
        cv2.imwrite(f"{output_dir}/color/{base}_color.png", 
                    cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    
    print(f"\n✓ Saved {len(test_dataset)} predictions to {output_dir}")

print("\n" + "="*50)
print("DONE")
print("="*50)
