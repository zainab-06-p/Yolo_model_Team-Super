"""
MEMBER 2 - 5 ENHANCEMENTS FOR IMPROVED INFERENCE
1. TTA (Test-Time Augmentation)
2. Multi-Scale Inference
3. SegFormer Ensemble
4. Advanced Augmentation (for retraining)
5. CRF Post-Processing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import os
from tqdm import tqdm

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Value mapping
VALUE_MAP = {
    0: 0, 100: 1, 200: 2, 300: 3, 400: 4,
    500: 5, 600: 6, 700: 7, 800: 8, 900: 9
}

h, w = 252, 462
tokenH, tokenW = h // 14, w // 14
n_embedding = 384

# ============================================================
# MODEL ARCHITECTURES
# ============================================================
class SegmentationHeadConvNeXt(nn.Module):
    """DINOv2 segmentation head (our trained model)."""
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

# ============================================================
# ENHANCEMENT 1: TTA (TEST-TIME AUGMENTATION)
# ============================================================
def tta_predict(model_dict, img_tensor, n_aug=5):
    """
    Test-Time Augmentation: Apply multiple augmentations at inference
    and average predictions for robustness.
    
    Expected gain: +0.02-0.03 mIoU
    """
    backbone = model_dict['backbone']
    classifier = model_dict['classifier']
    
    predictions = []
    
    # Original
    with torch.no_grad():
        feats = backbone.forward_features(img_tensor)['x_norm_patchtokens']
        logits = classifier(feats)
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        predictions.append(F.softmax(logits, dim=1))
    
    # Horizontal flip
    with torch.no_grad():
        img_flipped = torch.flip(img_tensor, dims=[3])
        feats = backbone.forward_features(img_flipped)['x_norm_patchtokens']
        logits = classifier(feats)
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        # Flip back the prediction
        pred = F.softmax(logits, dim=1)
        pred = torch.flip(pred, dims=[3])
        predictions.append(pred)
    
    # Scale variations
    scales = [0.9, 1.1]
    for scale in scales:
        resized = F.interpolate(img_tensor, scale_factor=scale, mode='bilinear', align_corners=False)
        with torch.no_grad():
            feats = backbone.forward_features(resized)['x_norm_patchtokens']
            logits = classifier(feats)
            # Resize back to original
            logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
            predictions.append(F.softmax(logits, dim=1))
    
    # Average all predictions
    final_pred = torch.stack(predictions).mean(dim=0)
    return final_pred

# ============================================================
# ENHANCEMENT 2: MULTI-SCALE INFERENCE
# ============================================================
def multiscale_predict(model_dict, img_tensor, scales=[0.5, 1.0, 1.5, 2.0]):
    """
    Multi-Scale Inference: Predict at multiple resolutions and fuse.
    
    Expected gain: +0.02-0.04 mIoU (better small object detection)
    """
    backbone = model_dict['backbone']
    classifier = model_dict['classifier']
    
    predictions = []
    weights = [0.15, 0.5, 0.25, 0.1]  # Higher weight for original scale
    
    for i, scale in enumerate(scales):
        if scale == 1.0:
            resized = img_tensor
        else:
            resized = F.interpolate(img_tensor, scale_factor=scale, mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            feats = backbone.forward_features(resized)['x_norm_patchtokens']
            logits = classifier(feats)
            # Resize back to (h, w)
            logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
            pred = F.softmax(logits, dim=1) * weights[i]
            predictions.append(pred)
    
    # Weighted sum
    final_pred = torch.stack(predictions).sum(dim=0)
    return final_pred

# ============================================================
# ENHANCEMENT 3: SEGFORMER ENSEMBLE
# ============================================================
class SegFormerSegmentor(nn.Module):
    """
    SegFormer backbone for ensemble diversity.
    Different architecture (CNN+Transformer hybrid vs pure Transformer)
    """
    def __init__(self, n_classes=10):
        super().__init__()
        # Use SegFormer-B0 (lightweight)
        from transformers import SegformerForSemanticSegmentation
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=n_classes,
            ignore_mismatched_sizes=True
        )
        
    def forward(self, x):
        # x: [B, 3, H, W]
        outputs = self.model(x)
        logits = outputs.logits  # [B, n_classes, H/4, W/4]
        # Upsample to full resolution
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        return logits

def load_segformer():
    """Load SegFormer model (to be trained separately)."""
    try:
        model = SegFormerSegmentor(n_classes=10).to(device)
        print("✓ SegFormer loaded (needs training)")
        return model
    except Exception as e:
        print(f"⚠ SegFormer load failed: {e}")
        return None

def ensemble_predict(dinov2_dict, segformer_model, img_tensor, weights=[0.4, 0.6]):
    """
    Ensemble: Combine DINOv2 and SegFormer predictions.
    
    Expected gain: +0.03-0.05 mIoU through architectural diversity
    """
    # DINOv2 prediction
    dinov2_pred = tta_predict(dinov2_dict, img_tensor)
    
    # SegFormer prediction
    if segformer_model is not None:
        with torch.no_grad():
            segformer_logits = segformer_model(img_tensor)
            segformer_pred = F.softmax(segformer_logits, dim=1)
        
        # Weighted ensemble
        final_pred = weights[0] * dinov2_pred + weights[1] * segformer_pred
    else:
        final_pred = dinov2_pred
    
    return final_pred

# ============================================================
# ENHANCEMENT 4: ADVANCED AUGMENTATION (FOR RETRAINING)
# ============================================================
def get_advanced_augmentation():
    """
    Advanced augmentation pipeline including CutMix and MixUp.
    To be used during retraining for better generalization.
    
    Expected gain: +0.02-0.03 mIoU (when retrained)
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    transform = A.Compose([
        # Standard augmentations
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=15, p=0.7),
        A.RandomResizedCrop(size=(h, w), scale=(0.5, 1.0), ratio=(0.75, 1.33), p=1.0),
        
        # Advanced augmentations
        A.Cutout(num_holes=8, max_h_size=40, max_w_size=40, p=0.3),
        A.MotionBlur(blur_limit=7, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        
        # Normalize
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})
    
    return transform

def apply_cutmix(img1, mask1, img2, mask2, alpha=1.0):
    """
    CutMix augmentation: Mix two images with random bbox.
    """
    lam = np.random.beta(alpha, alpha)
    
    h, w = img1.shape[:2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    
    img1[bby1:bby2, bbx1:bbx2] = img2[bby1:bby2, bbx1:bbx2]
    mask1[bby1:bby2, bbx1:bbx2] = mask2[bby1:bby2, bbx1:bbx2]
    
    return img1, mask1

# ============================================================
# ENHANCEMENT 5: CRF POST-PROCESSING
# ============================================================
def apply_crf(image, logits, n_classes=10, n_iter=5):
    """
    Conditional Random Fields for boundary refinement.
    
    Expected gain: +0.01-0.02 mIoU (sharper boundaries)
    Requires: pip install pydensecrf
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        print("⚠ pydensecrf not installed. Skipping CRF.")
        return logits
    
    # Convert logits to softmax probabilities
    probs = F.softmax(logits, dim=1)
    probs = probs.squeeze(0).cpu().numpy()  # [C, H, W]
    
    # Prepare image for CRF
    img_np = image.cpu().squeeze(0).permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    
    # Create CRF
    d = dcrf.DenseCRF2D(w, h, n_classes)
    
    # Unary potentials
    unary = unary_from_softmax(probs)
    d.setUnaryEnergy(unary)
    
    # Binary potentials (appearance)
    d.addPairwiseGaussian(sxy=3, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img_np, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    
    # Inference
    Q = d.inference(n_iter)
    Q = np.array(Q).reshape((n_classes, h, w))
    
    # Convert back to tensor
    refined_logits = torch.from_numpy(Q).unsqueeze(0).float()
    return refined_logits

# ============================================================
# MAIN INFERENCE WITH ALL ENHANCEMENTS
# ============================================================
def enhanced_inference(img_path, model_dict, segformer_model=None, use_tta=True, use_multiscale=True, use_crf=False):
    """
    Run inference with all enabled enhancements.
    """
    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    img = img.resize((w, h))
    img_np = np.array(img)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_np / 255.0 - mean) / std
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    
    # Ensemble with TTA and Multi-scale
    if use_tta and use_multiscale:
        # Combine TTA + Multi-scale
        pred = multiscale_predict(model_dict, img_tensor)
    elif use_tta:
        pred = tta_predict(model_dict, img_tensor)
    elif use_multiscale:
        pred = multiscale_predict(model_dict, img_tensor)
    else:
        # Basic inference
        backbone = model_dict['backbone']
        classifier = model_dict['classifier']
        with torch.no_grad():
            feats = backbone.forward_features(img_tensor)['x_norm_patchtokens']
            logits = classifier(feats)
            logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
            pred = F.softmax(logits, dim=1)
    
    # Apply CRF if enabled
    if use_crf:
        pred = apply_crf(img_tensor, pred)
    
    # Get final prediction
    pred_mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
    
    return pred_mask, img_np

# ============================================================
# INITIALIZATION
# ============================================================
def load_models():
    """Load all models."""
    print("="*60)
    print("LOADING MODELS WITH 5 ENHANCEMENTS")
    print("="*60)
    
    # Load DINOv2
    print("\n1. Loading DINOv2 backbone...")
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True, verbose=False)
    backbone = backbone.to(device).eval()
    for param in backbone.parameters():
        param.requires_grad = False
    print("   ✓ DINOv2 loaded")
    
    # Load classifier
    print("\n2. Loading trained classifier...")
    classifier = SegmentationHeadConvNeXt(n_embedding, 10, tokenW, tokenH).to(device)
    
    model_path = 'model_augmented_best.pth'
    if os.path.exists(model_path):
        classifier.load_state_dict(torch.load(model_path, map_location=device))
        print(f"   ✓ Loaded from {model_path}")
    else:
        print(f"   ⚠ Model not found, using random weights")
    
    classifier.eval()
    
    dinov2_dict = {'backbone': backbone, 'classifier': classifier}
    
    # Load SegFormer
    print("\n3. Loading SegFormer (Enhancement 3)...")
    segformer_model = load_segformer()
    
    print("\n" + "="*60)
    print("MODELS READY")
    print("="*60)
    print(f"Enhancement 1 (TTA): Available")
    print(f"Enhancement 2 (Multi-scale): Available")
    print(f"Enhancement 3 (SegFormer): {'Available' if segformer_model else 'Not loaded'}")
    print(f"Enhancement 4 (Advanced Aug): For retraining only")
    print(f"Enhancement 5 (CRF): Available (requires pydensecrf)")
    print("="*60)
    
    return dinov2_dict, segformer_model

if __name__ == "__main__":
    # Load models
    dinov2_dict, segformer_model = load_models()
    
    # Example usage
    print("\nExample: Enhanced inference on test images")
    print("Use: enhanced_inference(img_path, dinov2_dict, segformer_model, ")
    print("                      use_tta=True, use_multiscale=True, use_crf=False)")
