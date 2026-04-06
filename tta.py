"""
tta.py — Test-Time Augmentation for Offroad Segmentation

Standalone TTA script for inference on validation or test sets.
Supports multiple TTA variants and ensemble averaging.

Usage:
  # Single model with TTA on validation set
  python tta.py --model_path model_augmented_best.pth --mode val --data_dir val --tta
  
  # Test set inference with TTA
  python tta.py --model_path model_finetuned_best.pth --mode test --data_dir test --tta --output_dir predictions
  
  # Without TTA (baseline)
  python tta.py --model_path model.pth --mode val --data_dir val

Supported TTA variants:
  - original (always included)
  - hflip (horizontal flip)
  - Note: Scale variants (0.75x, 1.25x) are DISABLED as they break token grid
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# ── Configuration ─────────────────────────────────────────────
IMG_H, IMG_W = 252, 462  # DINOv2-vits14: multiples of 14
TOKEN_H = IMG_H // 14    # 18
TOKEN_W = IMG_W // 14    # 33
N_CLASSES = 10
N_EMB = 384              # dinov2_vits14 embedding dimension

VALUE_MAP = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

COLOR_PALETTE = np.array([
    [0,   0,   0  ],   # 0 Background
    [34,  139, 34 ],   # 1 Trees
    [0,   255, 0  ],   # 2 Lush Bushes
    [210, 180, 140],    # 3 Dry Grass
    [139, 90,  43 ],    # 4 Dry Bushes
    [128, 128, 0  ],    # 5 Ground Clutter
    [139, 69,  19 ],     # 6 Logs
    [128, 128, 128],    # 7 Rocks
    [160, 82,  45 ],    # 8 Landscape
    [135, 206, 235],    # 9 Sky
], dtype=np.uint8)


# ── Model Architecture ────────────────────────────────────────
class SegmentationHeadConvNeXt(nn.Module):
    """ConvNeXt-style segmentation head."""
    def __init__(self, in_channels, out_channels, tokenW, tokenH, dropout=0.1):
        super().__init__()
        self.tokenH, self.tokenW = tokenH, tokenW
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=7, padding=3),
            nn.GELU(),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.dropout = nn.Dropout2d(p=dropout)
        self.classifier = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        """x: [B, N, C] -> [B, n_classes, tokenH, tokenW]"""
        B, N, C = x.shape
        x = x.reshape(B, self.tokenH, self.tokenW, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.dropout(x)
        return self.classifier(x)


# ── Dataset Classes ────────────────────────────────────────────
class ValDataset(Dataset):
    """Validation dataset with ground truth."""
    def __init__(self, data_dir):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.mask_dir = os.path.join(data_dir, 'Segmentation')
        self.filenames = sorted(os.listdir(self.image_dir))
        
        self.img_transform = transforms.Compose([
            transforms.Resize((IMG_H, IMG_W)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def convert_mask(self, mask_pil):
        arr = np.array(mask_pil)
        out = np.zeros_like(arr, dtype=np.uint8)
        for raw_val, class_id in VALUE_MAP.items():
            out[arr == raw_val] = class_id
        return out

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = self.img_transform(Image.open(os.path.join(self.image_dir, fname)).convert('RGB'))
        mask_pil = Image.open(os.path.join(self.mask_dir, fname))
        mask = self.convert_mask(mask_pil)
        return img, mask, fname


class TestDataset(Dataset):
    """Test dataset without ground truth."""
    def __init__(self, data_dir):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.filenames = sorted(os.listdir(self.image_dir))
        
        self.img_transform = transforms.Compose([
            transforms.Resize((IMG_H, IMG_W)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = self.img_transform(Image.open(os.path.join(self.image_dir, fname)).convert('RGB'))
        return img, fname


# ── TTA Functions ─────────────────────────────────────────────
def tta_hflip(img_tensor):
    """Apply horizontal flip."""
    return torch.flip(img_tensor, dims=[-1])


def undo_hflip(pred_tensor):
    """Undo horizontal flip on prediction."""
    return torch.flip(pred_tensor, dims=[-1])


def apply_tta(img, variant='original'):
    """Apply TTA variant to image."""
    if variant == 'original':
        return img, lambda x: x
    elif variant == 'hflip':
        return tta_hflip(img), undo_hflip
    else:
        raise ValueError(f"Unknown TTA variant: {variant}")


@torch.no_grad()
def predict_with_tta(model, backbone, img, use_tta=True, device='cuda'):
    """
    Run inference with optional TTA.
    
    Args:
        model: Segmentation head
        backbone: DINOv2 backbone
        img: Input image tensor [B, 3, H, W]
        use_tta: Whether to use TTA
    
    Returns:
        probs: Softmax probabilities [B, C, H, W]
    """
    B, _, H, W = img.shape
    img = img.to(device)
    
    if not use_tta:
        # Single forward pass
        feats = backbone.forward_features(img)['x_norm_patchtokens']
        logits = model(feats)
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        return F.softmax(logits, dim=1)
    
    # TTA: original + hflip
    variants = ['original', 'hflip']
    all_probs = []
    
    for variant in variants:
        aug_img, undo_fn = apply_tta(img, variant)
        
        feats = backbone.forward_features(aug_img)['x_norm_patchtokens']
        logits = model(feats)
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        probs = F.softmax(logits, dim=1)
        
        # Undo augmentation if needed
        if variant != 'original':
            probs = undo_fn(probs)
        
        all_probs.append(probs)
    
    # Average probabilities
    return torch.stack(all_probs).mean(dim=0)


# ── Metrics ───────────────────────────────────────────────────
def compute_miou(pred, target, n_classes=10):
    """Compute mean IoU."""
    pred = pred.cpu().numpy().flatten()
    target = target.flatten()
    
    ious = []
    for c in range(n_classes):
        pred_c = pred == c
        target_c = target == c
        inter = np.logical_and(pred_c, target_c).sum()
        union = np.logical_or(pred_c, target_c).sum()
        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(inter / union)
    
    return np.nanmean(ious), ious


def print_iou_table(mean_iou, iou_list):
    """Print formatted IoU table."""
    print(f"\n{'='*60}")
    print(f"  Mean IoU: {mean_iou:.4f}")
    print(f"{'='*60}")
    for i, (name, iou) in enumerate(zip(CLASS_NAMES, iou_list)):
        val_str = f"{iou:.4f}" if not np.isnan(iou) else "  N/A"
        bar = '█' * int((iou if not np.isnan(iou) else 0) * 20)
        rare = " ← RARE" if i in [5, 6] else ""
        print(f"  [{i}] {name:<16}: {val_str}  {bar}{rare}")
    print(f"{'='*60}")


# ── Main Inference ─────────────────────────────────────────────
def run_inference(model, backbone, dataloader, mode='val', use_tta=False, 
                  output_dir=None, device='cuda'):
    """
    Run inference on validation or test set.
    
    Args:
        model: Segmentation head
        backbone: DINOv2 backbone
        dataloader: DataLoader
        mode: 'val' or 'test'
        use_tta: Whether to use TTA
        output_dir: Directory to save predictions (for test mode)
        device: Device to run on
    """
    model.eval()
    backbone.eval()
    
    all_ious = []
    all_class_ious = []
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'masks_color'), exist_ok=True)
    
    saved_count = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Inference (TTA={use_tta})")
        
        for batch in pbar:
            if mode == 'val':
                imgs, masks, fnames = batch
                masks = masks.numpy()
            else:
                imgs, fnames = batch
            
            # Run TTA inference
            probs = predict_with_tta(model, backbone, imgs, use_tta, device)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            
            if mode == 'val':
                # Compute metrics
                for i in range(len(preds)):
                    miou, class_ious = compute_miou(torch.from_numpy(preds[i]), masks[i])
                    all_ious.append(miou)
                    all_class_ious.append(class_ious)
                
                pbar.set_postfix(mIoU=np.nanmean(all_ious))
            
            # Save predictions
            if output_dir:
                for i, fname in enumerate(fnames):
                    base = Path(fname).stem
                    
                    # Raw mask
                    Image.fromarray(preds[i].astype(np.uint8)).save(
                        os.path.join(output_dir, 'masks', f"{base}_pred.png"))
                    
                    # Colorized mask
                    color_mask = COLOR_PALETTE[preds[i]]
                    cv2.imwrite(
                        os.path.join(output_dir, 'masks_color', f"{base}_color.png"),
                        cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
                    
                    saved_count += 1
    
    # Print results
    if mode == 'val' and all_ious:
        mean_iou = np.nanmean(all_ious)
        mean_class_iou = np.nanmean(all_class_ious, axis=0)
        print_iou_table(mean_iou, mean_class_iou)
        return mean_iou, mean_class_iou
    
    if mode == 'test':
        print(f"\n✓ Saved {saved_count} predictions to {output_dir}")
    
    return None, None


# ── Main Entry Point ───────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Test-Time Augmentation Inference')
    parser.add_argument('--model_path', required=True, help='Path to model weights')
    parser.add_argument('--mode', choices=['val', 'test'], required=True,
                       help='Validation or test mode')
    parser.add_argument('--data_dir', required=True, help='Path to data directory')
    parser.add_argument('--output_dir', default='tta_predictions', 
                       help='Output directory for predictions')
    parser.add_argument('--tta', action='store_true', help='Enable TTA')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--device', default='cuda', help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Validate device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"\n{'='*60}")
    print(f"  TTA Inference")
    print(f"{'='*60}")
    print(f"  Model: {args.model_path}")
    print(f"  Mode: {args.mode}")
    print(f"  TTA: {'ON (2 variants)' if args.tta else 'OFF'}")
    print(f"  Device: {args.device}")
    print(f"{'='*60}\n")
    
    # Load model
    print("Loading DINOv2 backbone...")
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14',
                              pretrained=True, verbose=False)
    backbone = backbone.to(args.device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    
    print("Loading segmentation head...")
    model = SegmentationHeadConvNeXt(N_EMB, N_CLASSES, TOKEN_W, TOKEN_H)
    
    # Load weights
    state = torch.load(args.model_path, map_location=args.device, weights_only=False)
    if isinstance(state, dict) and 'classifier_state' in state:
        state = state['classifier_state']
    model.load_state_dict(state, strict=True)
    model = model.to(args.device).eval()
    
    print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} params\n")
    
    # Create dataset
    if args.mode == 'val':
        dataset = ValDataset(args.data_dir)
    else:
        dataset = TestDataset(args.data_dir)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Dataset: {len(dataset)} images")
    print(f"Batch size: {args.batch_size}\n")
    
    # Run inference
    start_time = time.time()
    mean_iou, class_ious = run_inference(
        model, backbone, dataloader, args.mode, args.tta, 
        args.output_dir if args.mode == 'test' else None, args.device
    )
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"  Inference complete in {elapsed:.1f}s ({elapsed/len(dataset):.2f}s per image)")
    print(f"{'='*60}")
    
    if args.mode == 'val' and mean_iou:
        improvement = "with TTA" if args.tta else "without TTA"
        print(f"\nFinal mIoU {improvement}: {mean_iou:.4f}")


if __name__ == '__main__':
    main()
