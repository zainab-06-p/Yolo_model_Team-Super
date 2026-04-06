"""
Step-by-Step Kaggle Training Guide
Train NEW Member 3 Model + Complete Ensemble
"""

# ═══════════════════════════════════════════════════════════════
# STEP 1: CREATE KAGGLE NOTEBOOK
# ═══════════════════════════════════════════════════════════════

"""
1. Go to https://www.kaggle.com
2. Click "Code" → "New Notebook"
3. On right side panel:
   - Accelerator: Select "GPU T4 x2"
   - Language: Python
4. Click "Create"
"""

# ═══════════════════════════════════════════════════════════════
# STEP 2: ADD YOUR DATASET
# ═══════════════════════════════════════════════════════════════

"""
1. In notebook, look for "Add Input" button (top right)
2. Click it → "Datasets" tab
3. Search: "yolo-training-data"
4. Click on your dataset → Click "Add"
5. The dataset will be at: /kaggle/input/yolo-training-data/
"""

# ═══════════════════════════════════════════════════════════════
# STEP 3: UPLOAD MODELS (M1 and M2)
# ═══════════════════════════════════════════════════════════════

"""
1. In Kaggle notebook, look for "+ Add Data" button
2. Click "Upload"
3. Upload these 2 files from your PC:
   - model_finetuned_best.pth.zip (or .pth)
   - model_augmented_best.pth
4. They will be at: /kaggle/working/

OR

1. Create a Kaggle Dataset with your models:
   - Go to kaggle.com → Datasets → New Dataset
   - Name: "my-ensemble-models"
   - Upload: M1 and M2 files
   - Click "Create"
   - Add this dataset to your notebook
   - Path: /kaggle/input/my-ensemble-models/
"""

# ═══════════════════════════════════════════════════════════════
# STEP 4: TRAIN NEW MEMBER 3 MODEL
# ═══════════════════════════════════════════════════════════════

"""
Copy this entire code into a NOTEBOOK CELL and run:
"""

import os
import numpy as np
from PIL import Image, ImageEnhance
import cv2
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Paths (adjust these based on your Kaggle setup)
TRAIN_DIR = '/kaggle/input/yolo-training-data/Offroad_Segmentation_Training_Dataset/train'
VAL_DIR = '/kaggle/input/yolo-training-data/Offroad_Segmentation_Training_Dataset/val'

# Image settings
IMG_H, IMG_W = 252, 462  # 18x33 tokens for vits14
TOKEN_H, TOKEN_W = 18, 33
N_CLASSES = 10

# Class mapping
VALUE_MAP = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

def convert_mask(mask_pil):
    arr = np.array(mask_pil)
    out = np.zeros_like(arr, dtype=np.uint8)
    for raw, cls in VALUE_MAP.items():
        out[arr == raw] = cls
    return out

# Model Architecture - Member 3 Style (Strong Augmentation)
class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.tokenH, self.tokenW = tokenH, tokenW
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 256, 7, padding=3),
            nn.GELU(),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(256, 256, 7, padding=3, groups=256),
            nn.GELU(),
            nn.Conv2d(256, 256, 1),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GELU(),
        )
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

# Data Augmentation (Strong - Member 3 Style)
class StrongAugment:
    def __call__(self, img, mask):
        img_np = np.array(img)
        mask_np = np.array(mask)
        
        # Horizontal flip
        if random.random() > 0.5:
            img_np = np.fliplr(img_np)
            mask_np = np.fliplr(mask_np)
        
        # Random rotation
        if random.random() > 0.7:
            angle = random.randint(-10, 10)
            h, w = img_np.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            img_np = cv2.warpAffine(img_np, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            mask_np = cv2.warpAffine(mask_np, M, (w, h), borderMode=cv2.BORDER_REFLECT, 
                                    flags=cv2.INTER_NEAREST)
        
        # Random crop and resize
        if random.random() > 0.7:
            h, w = img_np.shape[:2]
            crop_h, crop_w = int(h*0.8), int(w*0.8)
            y = random.randint(0, h - crop_h)
            x = random.randint(0, w - crop_w)
            img_np = img_np[y:y+crop_h, x:x+crop_w]
            mask_np = mask_np[y:y+crop_h, x:x+crop_w]
            img_np = cv2.resize(img_np, (w, h))
            mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Color jitter
        if random.random() > 0.5:
            img_pil = Image.fromarray(img_np)
            enhancer = ImageEnhance.Color(img_pil)
            img_pil = enhancer.enhance(random.uniform(0.8, 1.2))
            enhancer = ImageEnhance.Brightness(img_pil)
            img_pil = enhancer.enhance(random.uniform(0.8, 1.2))
            img_np = np.array(img_pil)
        
        # Add noise
        if random.random() > 0.8:
            noise = np.random.normal(0, 10, img_np.shape).astype(np.int16)
            img_np = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_np), Image.fromarray(mask_np)

# Dataset
img_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

mask_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

class DesertDataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.mask_dir = os.path.join(data_dir, 'Segmentation')
        self.filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        self.is_train = is_train
        self.augment = StrongAugment() if is_train else None
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(os.path.join(self.image_dir, fname)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, fname))
        
        if self.is_train and self.augment:
            img, mask = self.augment(img, mask)
        
        mask_cls = Image.fromarray(convert_mask(mask))
        
        img_t = img_transform(img)
        mask_t = (mask_transform(mask_cls) * 255).long().squeeze(0)
        
        return img_t, mask_t

# Load backbone
print("\nLoading DINOv2 backbone...")
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', 
                          pretrained=True, verbose=False)
backbone = backbone.to(DEVICE).eval()
for p in backbone.parameters():
    p.requires_grad = False

# Detect embedding dimension
with torch.no_grad():
    probe = torch.zeros(1, 3, IMG_H, IMG_W, device=DEVICE)
    feat = backbone.forward_features(probe)['x_norm_patchtokens']
    N_EMB = feat.shape[2]
    print(f"Embed dim: {N_EMB}")

# Create model
model = SegmentationHeadConvNeXt(N_EMB, N_CLASSES, TOKEN_W, TOKEN_H).to(DEVICE)

# Count parameters
total = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total:,}")

# Datasets
train_dataset = DesertDataset(TRAIN_DIR, is_train=True)
val_dataset = DesertDataset(VAL_DIR, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = OneCycleLR(optimizer, max_lr=1e-4, epochs=50, 
                       steps_per_epoch=len(train_loader))

# Training loop
print("\n" + "="*50)
print("TRAINING NEW MEMBER 3 MODEL")
print("="*50)

best_miou = 0.0

for epoch in range(50):
    # Train
    model.train()
    train_loss = 0.0
    
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/50"):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        
        optimizer.zero_grad()
        
        with torch.no_grad():
            features = backbone.forward_features(images)['x_norm_patchtokens']
        
        outputs = model(features)
        outputs = F.interpolate(outputs, size=(IMG_H, IMG_W), mode='bilinear', align_corners=False)
        
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
    
    # Validate
    model.eval()
    val_ious = []
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            features = backbone.forward_features(images)['x_norm_patchtokens']
            outputs = model(features)
            outputs = F.interpolate(outputs, size=(IMG_H, IMG_W), mode='bilinear', align_corners=False)
            
            preds = outputs.argmax(dim=1)
            
            for c in range(N_CLASSES):
                pred_c = (preds == c)
                target_c = (masks == c)
                inter = (pred_c & target_c).sum().float()
                union = (pred_c | target_c).sum().float()
                if union > 0:
                    val_ious.append((inter/union).item())
    
    mean_iou = np.mean(val_ious) if val_ious else 0
    
    print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Val mIoU={mean_iou:.4f}")
    
    if mean_iou > best_miou:
        best_miou = mean_iou
        torch.save(model.state_dict(), '/kaggle/working/model_member3_new.pth')
        print(f"  ✓ New best model saved! (mIoU={best_miou:.4f})")

print(f"\nTraining complete! Best mIoU: {best_miou:.4f}")
print("Model saved to: /kaggle/working/model_member3_new.pth")

# ═══════════════════════════════════════════════════════════════
# STEP 5: VERIFY TRAINED MODEL
# ═══════════════════════════════════════════════════════════════

"""
After training completes:

1. Check output files:
   !ls -lh /kaggle/working/*.pth

2. You should see:
   model_member3_new.pth

3. Download this file:
   - Click on file in right panel
   - Click "Download"
   - Save to your PC
"""

# ═══════════════════════════════════════════════════════════════
# STEP 6: CREATE ENSEMBLE WITH ALL 3 MODELS
# ═══════════════════════════════════════════════════════════════

"""
Create NEW notebook cell with ensemble code:
"""

# Copy this code to a new cell:

import os
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

# Model paths
MEMBER1_PATH = '/kaggle/input/yolo-training-data/model_finetuned_best.pth'  # Update path
MEMBER2_PATH = '/kaggle/input/yolo-training-data/model_augmented_best.pth'   # Update path
MEMBER3_PATH = '/kaggle/working/model_member3_new.pth'  # Your newly trained model

# Data paths
DATASET_PATH = '/kaggle/input/yolo-training-data'

# Image settings
IMG_H, IMG_W = 252, 462
TOKEN_H, TOKEN_W = 18, 33
N_CLASSES = 10
N_EMB = 384

# Ensemble weights (tune based on validation performance)
ENSEMBLE_WEIGHTS = {
    'M1': 0.40,  # Member 1 - Fine-tuned
    'M2': 0.30,  # Member 2 - Augmented
    'M3': 0.30,  # Member 3 - New (your trained model)
}

# Class mapping
VALUE_MAP = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

# Model Architecture
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

# Load helpers
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
        print(f"⚠ {name}: Not found at {ckpt_path}")
        return None
    
    resolved = extract_zip(ckpt_path)
    head = SegmentationHeadConvNeXt(N_EMB, N_CLASSES, TOKEN_W, TOKEN_H).to(device)
    
    try:
        state = torch.load(resolved, map_location=device)
        if isinstance(state, dict):
            if 'classifier_state' in state:
                state = state['classifier_state']
            elif 'state_dict' in state:
                state = state['state_dict']
        
        head.load_state_dict(state, strict=True)
        head.eval()
        print(f"✓ {name}: Loaded from {ckpt_path}")
        return head
    except Exception as e:
        print(f"⚠ {name}: Load failed - {e}")
        return None

# Load backbone
print("\nLoading DINOv2...")
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', 
                          pretrained=True, verbose=False)
backbone = backbone.to(device).eval()
for p in backbone.parameters():
    p.requires_grad = False
print("✓ Backbone loaded\n")

# Load all 3 models
models = {}
for name, path in [('M1', MEMBER1_PATH), ('M2', MEMBER2_PATH), ('M3', MEMBER3_PATH)]:
    model = load_head(path, f"Member {name}")
    if model:
        models[name] = model

if len(models) < 3:
    print(f"\n⚠ Warning: Only {len(models)}/3 models loaded")
    print("Check your paths above!")
else:
    print(f"\n✓ All 3 models loaded successfully!")

# Transforms
img_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Inference function
@torch.no_grad()
def predict_ensemble(img_tensor, use_tta=True):
    img_tensor = img_tensor.to(device)
    total_weight = sum(ENSEMBLE_WEIGHTS[name] for name in models.keys())
    combined = torch.zeros(N_CLASSES, IMG_H, IMG_W, device=device)
    
    for name, head in models.items():
        weight = ENSEMBLE_WEIGHTS[name]
        
        # Original
        feats = backbone.forward_features(img_tensor.unsqueeze(0))['x_norm_patchtokens']
        logits = head(feats)
        logits = F.interpolate(logits, size=(IMG_H, IMG_W), mode='bilinear', align_corners=False)
        probs = F.softmax(logits, dim=1).squeeze(0)
        
        if use_tta:
            # Horizontal flip
            img_flip = torch.flip(img_tensor, dims=[-1])
            feats_flip = backbone.forward_features(img_flip.unsqueeze(0))['x_norm_patchtokens']
            logits_flip = head(feats_flip)
            logits_flip = F.interpolate(logits_flip, size=(IMG_H, IMG_W), mode='bilinear', align_corners=False)
            probs_flip = F.softmax(logits_flip, dim=1).squeeze(0)
            probs_flip = torch.flip(probs_flip, dims=[-1])
            probs = (probs + probs_flip) / 2
        
        combined += weight * probs
    
    combined /= total_weight
    return torch.argmax(combined, dim=0).cpu().numpy()

# ═══════════════════════════════════════════════════════════════
# STEP 7: RUN VALIDATION
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
    return np.nanmean(ious)

# Validate
print("\n" + "="*50)
print("VALIDATING 3-MODEL ENSEMBLE")
print("="*50)

val_image_dir = os.path.join(DATASET_PATH, 'Offroad_Segmentation_Training_Dataset/val/Color_Images')
val_mask_dir = os.path.join(DATASET_PATH, 'Offroad_Segmentation_Training_Dataset/val/Segmentation')

val_images = sorted([f for f in os.listdir(val_image_dir) if f.endswith('.png')])

ious = []
for fname in tqdm(val_images[:50], desc="Validating"):  # Test on 50 images
    img_path = os.path.join(val_image_dir, fname)
    mask_path = os.path.join(val_mask_dir, fname)
    
    if not os.path.exists(mask_path):
        continue
    
    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path)
    
    img_tensor = img_transform(img)
    mask_np = convert_mask(mask)
    mask_resized = Image.fromarray(mask_np).resize((IMG_W, IMG_H), Image.NEAREST)
    mask_np_resized = np.array(mask_resized)
    
    pred = predict_ensemble(img_tensor, use_tta=True)
    miou = compute_iou(pred, mask_np_resized)
    ious.append(miou)

mean_miou = np.nanmean(ious)
print(f"\n{'='*50}")
print(f"3-MODEL ENSEMBLE RESULT")
print(f"{'='*50}")
print(f"Mean mIoU: {mean_miou:.4f}")
print(f"Target: >= 0.50")
print(f"{'='*50}")

# ═══════════════════════════════════════════════════════════════
# STEP 8: GENERATE TEST PREDICTIONS
# ═══════════════════════════════════════════════════════════════

print("\nGenerating test predictions...")

test_dir = os.path.join(DATASET_PATH, 'Offroad_Segmentation_testImages/Color_Images')
output_dir = '/kaggle/working/final_predictions'
os.makedirs(output_dir, exist_ok=True)

if os.path.exists(test_dir):
    test_images = sorted([f for f in os.listdir(test_dir) if f.endswith('.png')])
    
    for fname in tqdm(test_images, desc="Test predictions"):
        img_path = os.path.join(test_dir, fname)
        img = Image.open(img_path).convert('RGB')
        img_tensor = img_transform(img)
        
        pred = predict_ensemble(img_tensor, use_tta=True)
        
        base = Path(fname).stem
        Image.fromarray(pred.astype(np.uint8)).save(f"{output_dir}/{base}_pred.png")
    
    print(f"✓ Saved {len(test_images)} predictions to {output_dir}")
else:
    print(f"⚠ Test directory not found: {test_dir}")

print("\n" + "="*50)
print("COMPLETE!")
print("="*50)
print(f"Final model: model_member3_new.pth")
print(f"Predictions: {output_dir}")
print(f"Expected mIoU: {mean_miou:.4f}")
