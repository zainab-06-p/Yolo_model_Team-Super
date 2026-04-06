"""
BONUS ROUND INFERENCE - Member 2 Model
Quick inference for Round 2 submission
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import os

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Class mapping (same as training)
VALUE_MAP = {
    0: 0, 100: 1, 200: 2, 300: 3, 400: 4,
    500: 5, 600: 6, 700: 7, 800: 8, 900: 9
}

# Color palette for visualization
COLOR_PALETTE = [
    [135, 206, 235],    # 0 Sky - Light Blue
    [210, 180, 140],    # 1 Ground - Tan
    [128, 128, 128],    # 2 Small Rocks - Gray
    [34, 139, 34],      # 3 Vegetation - Forest Green
    [105, 105, 105],    # 4 Large Rocks - Dark Gray
    [139, 69, 19],      # 5 Ground Clutter - Brown
    [160, 82, 45],      # 6 Logs - Sienna
    [255, 255, 0],      # 7 Poles - Yellow
    [128, 0, 128],      # 8 Fences - Purple
    [255, 0, 0],        # 9 Sign - Red
]

def convert_mask_to_classes(mask):
    """Convert 16-bit mask to class IDs."""
    out = np.zeros_like(mask, dtype=np.uint8)
    for raw_val, class_id in VALUE_MAP.items():
        out[mask == raw_val] = class_id
    return out

def mask_to_color(mask, palette=COLOR_PALETTE):
    """Convert class mask to color image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(palette):
        color_mask[mask == i] = color
    return color_mask

# Model architecture (same as training)
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
        return self.classifier(self.dropout(x))

# Load model
print("Loading model...")
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True, verbose=False)
backbone = backbone.to(device).eval()
for param in backbone.parameters():
    param.requires_grad = False

# Get dimensions
h, w = 252, 462
tokenH, tokenW = h // 14, w // 14
n_embedding = 384  # DINOv2-S dimension

classifier = SegmentationHeadConvNeXt(n_embedding, 10, tokenW, tokenH).to(device)

# Load trained weights
model_path = 'model_augmented_best.pth'
if os.path.exists(model_path):
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✓ Loaded model from {model_path}")
else:
    print(f"⚠ Model file not found: {model_path}")
    print("  Using untrained model (for testing)")

classifier.eval()

def preprocess_image(img_path):
    """Load and preprocess image."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((w, h))
    img_np = np.array(img)
    
    # Normalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_np / 255.0 - mean) / std
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)
    return img_tensor.to(device), img_np

def predict(img_path):
    """Run inference on single image."""
    img_tensor, img_orig = preprocess_image(img_path)
    
    with torch.no_grad():
        feats = backbone.forward_features(img_tensor)['x_norm_patchtokens']
        logits = classifier(feats)
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    
    return pred, img_orig

def compute_iou(pred_mask, gt_mask, n_classes=10):
    """Compute mIoU between prediction and ground truth."""
    ious = []
    for c in range(n_classes):
        pred_c = (pred_mask == c)
        gt_c = (gt_mask == c)
        intersection = np.logical_and(pred_c, gt_c).sum()
        union = np.logical_or(pred_c, gt_c).sum()
        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(intersection / union)
    return np.nanmean(ious), ious

def compute_pixel_accuracy(pred_mask, gt_mask):
    """Compute pixel-wise accuracy."""
    return (pred_mask == gt_mask).mean()

# Process images
print("\n" + "="*60)
print("BONUS ROUND INFERENCE")
print("="*60)

results = {}
base_dir = "round 2"

for img_num in [2, 3]:
    print(f"\n--- Processing Image {img_num} ---")
    
    # Paths
    before_path = os.path.join(base_dir, "before", f"{img_num}.jpg")
    after_path = os.path.join(base_dir, "after", f"{img_num} after.jpg")
    
    # Check files exist
    if not os.path.exists(before_path):
        print(f"⚠ Missing: {before_path}")
        continue
    if not os.path.exists(after_path):
        print(f"⚠ Missing: {after_path}")
        continue
    
    # Run prediction
    print(f"Running inference on image {img_num}...")
    pred_mask, orig_img = predict(before_path)
    
    # Load ground truth
    gt_img = Image.open(after_path)
    gt_np = np.array(gt_img)
    
    # Convert ground truth to class IDs (assuming it's colored, need to map back)
    # If ground truth is already class IDs, use directly
    if len(gt_np.shape) == 3:
        # It's RGB - we need to map colors to classes
        # Simplified: use first channel as proxy (may need adjustment)
        gt_mask = gt_np[:, :, 0] // 25  # Rough mapping (adjust based on actual colors)
        gt_mask = np.clip(gt_mask, 0, 9)
    else:
        gt_mask = gt_np
    
    # Resize gt to match pred if needed
    if gt_mask.shape != pred_mask.shape:
        gt_mask = cv2.resize(gt_mask.astype(np.uint8), (pred_mask.shape[1], pred_mask.shape[0]), 
                            interpolation=cv2.INTER_NEAREST)
    
    # Compute metrics
    mIoU, per_class_iou = compute_iou(pred_mask, gt_mask)
    pixel_acc = compute_pixel_accuracy(pred_mask, gt_mask)
    
    results[img_num] = {
        'mIoU': mIoU,
        'pixel_acc': pixel_acc,
        'per_class_iou': per_class_iou
    }
    
    print(f"  Image {img_num} mIoU: {mIoU:.4f}")
    print(f"  Image {img_num} Pixel Accuracy: {pixel_acc:.4f}")
    
    # Save prediction visualization
    pred_color = mask_to_color(pred_mask)
    gt_color = gt_img.resize((w, h)) if len(gt_img.size) == 2 else gt_img
    
    # Create side-by-side comparison
    fig = np.zeros((h, w*3, 3), dtype=np.uint8)
    fig[:, :w] = orig_img
    fig[:, w:2*w] = pred_color
    if isinstance(gt_color, Image.Image):
        fig[:, 2*w:] = np.array(gt_color.resize((w, h)))
    else:
        fig[:, 2*w:] = mask_to_color(gt_mask)
    
    output_path = os.path.join(base_dir, f"submission_{img_num}.jpg")
    Image.fromarray(fig).save(output_path)
    print(f"  Saved submission image: {output_path}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
avg_miou = np.mean([r['mIoU'] for r in results.values()])
avg_acc = np.mean([r['pixel_acc'] for r in results.values()])

print(f"\nAverage mIoU: {avg_miou:.4f}")
print(f"Average Pixel Accuracy: {avg_acc:.4f}")
print(f"\nOverall Accuracy Score: {avg_miou:.2%}")

print("\n" + "="*60)
print("SUBMISSION READY")
print("="*60)
print(f"Files created in: {base_dir}/")
print(f"  - submission_2.jpg (Before | Prediction | Ground Truth)")
print(f"  - submission_3.jpg (Before | Prediction | Ground Truth)")
print("\nUpload these to your team folder for Bonus Round!")
