"""
ROUND 2 - Simple Base DINOv2 Inference
Quick and reliable for bonus round submission
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os

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

# Get feature dimensions
with torch.no_grad():
    sample = torch.zeros(1, 3, h, w).to(device)
    feats = backbone.forward_features(sample)['x_norm_patchtokens']
    n_embedding = feats.shape[2]
    tokenH, tokenW = h // 14, w // 14  # 18, 33
    print(f"Features: {tokenH}x{tokenW}x{n_embedding}")

# Simple head that works with fixed dimensions
class SimpleSegHead(nn.Module):
    def __init__(self, in_dim, n_classes, th, tw):
        super().__init__()
        self.tokenH = th
        self.tokenW = tw
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )
    def forward(self, x):
        B, N, C = x.shape
        logits = self.head(x)  # [B, N, 10]
        # N should be tokenH * tokenW = 594
        logits = logits.permute(0, 2, 1).reshape(B, -1, self.tokenH, self.tokenW)
        return logits

seg_head = SimpleSegHead(n_embedding, 10, tokenH, tokenW).to(device).eval()

# Color palette
PALETTE = [
    [135, 206, 235], [210, 180, 140], [128, 128, 128], [34, 139, 34],
    [105, 105, 105], [139, 69, 19], [160, 82, 45], [255, 255, 0],
    [128, 0, 128], [255, 0, 0]
]

def mask_to_color(mask):
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, c in enumerate(PALETTE):
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

# TTA Prediction
def predict_with_tta(img_tensor):
    """Simple TTA: original + flip."""
    preds = []
    
    # Original
    with torch.no_grad():
        feats = backbone.forward_features(img_tensor)['x_norm_patchtokens']
        logits = seg_head(feats)
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        preds.append(F.softmax(logits, dim=1))
    
    # Horizontal flip
    with torch.no_grad():
        flipped = torch.flip(img_tensor, dims=[3])
        feats = backbone.forward_features(flipped)['x_norm_patchtokens']
        logits = seg_head(feats)
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        pred = torch.flip(logits, dims=[3])
        preds.append(F.softmax(pred, dim=1))
    
    # Average
    final = torch.stack(preds).mean(dim=0)
    return torch.argmax(final, dim=1).squeeze(0).cpu().numpy()

# Process images
print("\n" + "="*60)
print("ROUND 2: BASE DINOv2 + TTA")
print("="*60)

base_dir = "round 2"

for img_num in [2, 3]:
    print(f"\n--- Image {img_num} ---")
    
    before_path = os.path.join(base_dir, "before", f"Image {img_num} Before.jpg")
    if not os.path.exists(before_path):
        print(f"⚠ Missing: {before_path}")
        continue
    
    img_tensor, img_np = preprocess(before_path)
    
    # Predict with TTA
    print("  Running TTA prediction...")
    pred_mask = predict_with_tta(img_tensor)
    pred_color = mask_to_color(pred_mask)
    
    # Save prediction only (for submission)
    pred_path = os.path.join(base_dir, f"Image {img_num} After.jpg")
    Image.fromarray(pred_color).save(pred_path)
    print(f"  ✓ Saved: {pred_path}")
    
    # Also save comparison
    comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
    comparison[:, :w] = img_np
    comparison[:, w:] = pred_color
    comp_path = os.path.join(base_dir, f"comparison_{img_num}.jpg")
    Image.fromarray(comparison).save(comp_path)

print("\n" + "="*60)
print("DONE! Use 'Image 2 After.jpg' and 'Image 3 After.jpg'")
print("for your bonus round submission.")
print("="*60)
