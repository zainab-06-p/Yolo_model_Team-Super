"""
ROUND 2 - BASE DINOv2 with TTA + MULTI-SCALE
Enhanced inference for better results
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import os
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

h, w = 252, 462

# Load base DINOv2
print("Loading base DINOv2...")
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True, verbose=False)
backbone = backbone.to(device).eval()
for param in backbone.parameters():
    param.requires_grad = False
print("✓ DINOv2 loaded")

with torch.no_grad():
    sample = torch.zeros(1, 3, h, w).to(device)
    feats = backbone.forward_features(sample)['x_norm_patchtokens']
    n_embedding = feats.shape[2]
    tokenH, tokenW = h // 14, w // 14
    print(f"Features: {tokenH}x{tokenW}x{n_embedding}")

# Simple segmentation head
class SimpleSegHead(nn.Module):
    """Lightweight segmentation head - handles dynamic token sizes."""
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )
        
    def forward(self, x):
        B, N, C = x.shape
        logits = self.head(x)  # [B, N, n_classes]
        # Reshape to spatial - N = H * W, need to find H and W
        # Infer from sqrt, keeping aspect ratio roughly 2:1 (w/h ≈ 462/252 ≈ 1.8)
        H = int(np.sqrt(N / 1.8))  # Approximate height
        W = N // H  # Width is remaining
        if H * W != N:
            # Adjust to make exact
            H = int(np.sqrt(N))
            W = N // H
            if H * W < N:
                W += 1
        logits = logits.permute(0, 2, 1).reshape(B, -1, H, W)
        return logits

seg_head = SimpleSegHead(n_embedding, 10).to(device).eval()

# Color palette
COLOR_PALETTE = [
    [135, 206, 235], [210, 180, 140], [128, 128, 128], [34, 139, 34],
    [105, 105, 105], [139, 69, 19], [160, 82, 45], [255, 255, 0],
    [128, 0, 128], [255, 0, 0]
]

def mask_to_color(mask):
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, c in enumerate(COLOR_PALETTE):
        color[mask == i] = c
    return color

def preprocess(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((w, h))
    img_np = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_np / 255.0 - mean) / std
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)
    return img_tensor.to(device), img_np

def base_predict(img_tensor):
    """Base DINOv2 prediction with simple head."""
    with torch.no_grad():
        feats = backbone.forward_features(img_tensor)['x_norm_patchtokens']
        logits = seg_head(feats)
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        return F.softmax(logits, dim=1)

# ============================================================
# ENHANCEMENT 1: TTA (Test-Time Augmentation)
# ============================================================
def tta_predict(img_tensor, flip=True, rot=True):
    """Test-time augmentation."""
    preds = [base_predict(img_tensor)]
    
    if flip:
        # Horizontal flip
        flipped = torch.flip(img_tensor, dims=[3])
        pred = base_predict(flipped)
        pred = torch.flip(pred, dims=[3])
        preds.append(pred)
    
    # Average
    return torch.stack(preds).mean(dim=0)

# ============================================================
# ENHANCEMENT 2: MULTI-SCALE
# ============================================================
def multiscale_predict(img_tensor, scales=[0.8, 1.0, 1.2]):
    """Multi-scale prediction with proper sizing."""
    preds = []
    weights = [0.25, 0.5, 0.25]
    
    for i, scale in enumerate(scales):
        if scale == 1.0:
            resized = img_tensor
        else:
            # Ensure dimensions divisible by 14
            new_h = int((h * scale) // 14 * 14)
            new_w = int((w * scale) // 14 * 14)
            resized = F.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        pred = base_predict(resized)
        if scale != 1.0:
            pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=False)
        
        preds.append(pred * weights[i])
    
    return torch.stack(preds).sum(dim=0)

# ============================================================
# ENHANCEMENT 3: TTA + MULTI-SCALE COMBINED
# ============================================================
def enhanced_predict(img_tensor):
    """Combine TTA + Multi-scale."""
    scales = [0.9, 1.0, 1.1]
    all_preds = []
    
    for scale in scales:
        if scale == 1.0:
            resized = img_tensor
        else:
            # Ensure divisible by 14
            new_h = int((h * scale) // 14 * 14)
            new_w = int((w * scale) // 14 * 14)
            resized = F.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        # TTA at this scale
        pred = tta_predict(resized)
        
        if scale != 1.0:
            pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=False)
        
        all_preds.append(pred)
    
    # Average all
    final = torch.stack(all_preds).mean(dim=0)
    return final

# ============================================================
# MAIN
# ============================================================
print("\n" + "="*60)
print("ROUND 2: BASE DINOv2 + TTA + MULTI-SCALE")
print("="*60)

base_dir = "round 2"

for img_num in [2, 3]:
    print(f"\n--- Image {img_num} ---")
    
    before_path = os.path.join(base_dir, "before", f"Image {img_num} Before.jpg")
    if not os.path.exists(before_path):
        print(f"⚠ Missing: {before_path}")
        continue
    
    img_tensor, img_np = preprocess(before_path)
    
    # ENHANCED PREDICTION
    print("  Running enhanced inference (TTA + Multi-scale)...")
    pred_prob = enhanced_predict(img_tensor)
    pred_mask = torch.argmax(pred_prob, dim=1).squeeze(0).cpu().numpy()
    
    # Convert to color
    pred_color = mask_to_color(pred_mask)
    
    # Create comparison
    comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
    comparison[:, :w] = img_np
    comparison[:, w:] = pred_color
    
    # Save
    out_path = os.path.join(base_dir, f"FINAL_Image_{img_num}_Enhanced.jpg")
    Image.fromarray(comparison).save(out_path)
    print(f"  ✓ Saved: {out_path}")
    
    # Save prediction only
    pred_path = os.path.join(base_dir, f"Image {img_num} After.jpg")
    Image.fromarray(pred_color).save(pred_path)
    print(f"  ✓ Prediction: {pred_path}")

print("\n" + "="*60)
print("DONE - Use 'Image 2 After.jpg' and 'Image 3 After.jpg'")
print("for your bonus round submission!")
print("="*60)
