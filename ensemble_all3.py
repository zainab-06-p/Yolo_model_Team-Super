# ╔══════════════════════════════════════════════════════════════════════╗
# ║  FINAL 3-MODEL ENSEMBLE                                             ║
# ║  Members: 1 (DINOv2 2-phase fine-tune)                             ║
# ║           2 (Albumentations augmentation + weighted loss)           ║
# ║           3 (Hyperparameter search — best: OneCycleLR lr=1e-4)     ║
# ║                                                                      ║
# ║  Shared backbone : dinov2_vits14  (frozen, eval, NOT saved)         ║
# ║  Shared head     : SegmentationHeadConvNeXt (same arch, 3 weights)  ║
# ║  Ensemble        : weighted average of softmax probs                 ║
# ║  TTA             : 2-variant (original + h-flip)                    ║
# ║                    ⚠ Scale variants DISABLED — they change the      ║
# ║                      token grid which breaks the fixed classifier    ║
# ║                                                                      ║
# ║  Expected checkpoint formats (plain state_dict for all 3):          ║
# ║    member1 → model_finetuned_best.pth  (torch.save(state_dict))    ║
# ║    member2 → model_augmented_best.pth  (torch.save(state_dict))    ║
# ║    member3 → exp3_onecycle_lr1e4/model_best.pth                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ============================================================
# CELL 1 — IMPORTS
# ============================================================
import os, cv2, random, zipfile
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# ── Reproducibility ──────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} | Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# CELL 2 — PATHS  ← UPDATE THESE OR USE AUTO-DETECTION
# ============================================================

# ═══════════════════════════════════════════════════════════
# OPTION 1: MANUAL PATH OVERRIDE (Recommended)
# Paste your exact .pth file paths here. These will be used
# instead of auto-detection. Set to None to use auto-detection.
# ═══════════════════════════════════════════════════════════

MANUAL_MEMBER1_PATH = None  # e.g., '/kaggle/input/model-finetuned-best/model_finetuned_best.pth'
MANUAL_MEMBER2_PATH = None  # e.g., '/kaggle/input/model-augmented-best/model_augmented_best.pth'
MANUAL_MEMBER3_PATH = None  # e.g., '/kaggle/input/model-best/model_best.pth'

# ═══════════════════════════════════════════════════════════
# OPTION 2: AUTO-DETECTION (Fallback)
# Only used if manual paths above are None
# ═══════════════════════════════════════════════════════════

DATASET_SLUG = "yolo-training-data"   # Change if your dataset slug is different
DATASET_ROOT = f"/kaggle/input/datasets/adiinamdar/{DATASET_SLUG}"

def find_weight_file(path_list, member_name):
    """Find first existing weight file from list of candidates."""
    for p in path_list:
        if p is None:
            continue
        # Try exact path
        if os.path.isfile(p):
            return p
        # Try with .zip extension
        if os.path.isfile(p + '.zip'):
            return p + '.zip'
        # If it's a directory, look for any .pth file inside
        if os.path.isdir(p):
            pths = list(Path(p).glob('*.pth')) + list(Path(p).rglob('*.pth'))
            if pths:
                return str(pths[0])
        # Try in working directory
        working_path = f'/kaggle/working/{os.path.basename(p)}'
        if os.path.isfile(working_path):
            return working_path
        if os.path.isfile(working_path + '.zip'):
            return working_path + '.zip'
    return None

# Auto-detection candidates (only used if manual paths are None)
AUTO_MEMBER1_CANDIDATES = [
    '/kaggle/input/datasets/adiinamdar/model-finetuned-best/model_finetuned_best.pth',
    '/kaggle/input/datasets/adiinamdar/model-finetuned-best/model_finetuned_best.pth.zip',
    '/kaggle/input/model-finetuned-best/model_finetuned_best.pth',
    '/kaggle/input/model-finetuned-best/model_finetuned_best.pth.zip',
    '/kaggle/input/datasets/adiinamdar/model-finetuned-best',
    '/kaggle/input/model_finetuned_best.pth',
    '/kaggle/working/model_finetuned_best.pth',
    '/kaggle/working/model_finetuned_best.pth.zip',
]

AUTO_MEMBER2_CANDIDATES = [
    '/kaggle/input/datasets/adiinamdar/model-augmented-best/model_augmented_best.pth',
    '/kaggle/input/model-augmented-best/model_augmented_best.pth',
    '/kaggle/input/datasets/adiinamdar/model-augmented-best',
    '/kaggle/input/model_augmented_best.pth',
    '/kaggle/working/model_augmented_best.pth',
]

AUTO_MEMBER3_CANDIDATES = [
    '/kaggle/input/datasets/adiinamdar/model-best/model_best.pth',
    '/kaggle/input/datasets/adiinamdar/model-best/model_best.pth.zip',
    '/kaggle/input/model-best/model_best.pth',
    '/kaggle/input/model-best/model_best.pth.zip',
    '/kaggle/input/datasets/adiinamdar/model-best',
    '/kaggle/input/model_best.pth',
    '/kaggle/working/model_best.pth',
    '/kaggle/working/model_best.pth.zip',
]

# Resolve final paths: manual takes priority, then auto-detect
MEMBER1_PATH = MANUAL_MEMBER1_PATH if MANUAL_MEMBER1_PATH else find_weight_file(AUTO_MEMBER1_CANDIDATES, "Member 1")
MEMBER2_PATH = MANUAL_MEMBER2_PATH if MANUAL_MEMBER2_PATH else find_weight_file(AUTO_MEMBER2_CANDIDATES, "Member 2")
MEMBER3_PATH = MANUAL_MEMBER3_PATH if MANUAL_MEMBER3_PATH else find_weight_file(AUTO_MEMBER3_CANDIDATES, "Member 3")

CONFIG = {
    # ── Data ────────────────────────────────────────────────
    'val_dir' : f"{DATASET_ROOT}/Offroad_Segmentation_Training_Dataset/val",
    'test_dir': f"{DATASET_ROOT}/Offroad_Segmentation_testImages",

    # ── Model weights ────────────────────────────────────────
    'member1_path': MEMBER1_PATH,
    'member2_path': MEMBER2_PATH,
    'member3_path': MEMBER3_PATH,

    # ── Image size — MUST be multiples of 14 for DINOv2 ─────
    'img_h': 252,   # 18 × 14
    'img_w': 462,   # 33 × 14
    'n_classes': 10,
    'batch_size': 4,
    'num_workers': 2,

    # ── Ensemble weights [M1, M2, M3] ────────────────────────
    'ensemble_weights': [0.40, 0.30, 0.30],

    # ── TTA — 2 variants only ───────────────────────────────
    'use_tta': True,

    # ── Output ──────────────────────────────────────────────
    'output_dir': '/kaggle/working/ensemble_outputs',
}

print("\n" + "="*60)
print("PATH DETECTION RESULTS")
print("="*60)
if MANUAL_MEMBER1_PATH:
    print(f"  Member 1: ✓ MANUAL → {MEMBER1_PATH}")
else:
    print(f"  Member 1: {'✓ FOUND' if MEMBER1_PATH else '✗ NOT FOUND'} → {MEMBER1_PATH if MEMBER1_PATH else 'Using auto-detect'}")
if MANUAL_MEMBER2_PATH:
    print(f"  Member 2: ✓ MANUAL → {MEMBER2_PATH}")
else:
    print(f"  Member 2: {'✓ FOUND' if MEMBER2_PATH else '✗ NOT FOUND'} → {MEMBER2_PATH if MEMBER2_PATH else 'Using auto-detect'}")
if MANUAL_MEMBER3_PATH:
    print(f"  Member 3: ✓ MANUAL → {MEMBER3_PATH}")
else:
    print(f"  Member 3: {'✓ FOUND' if MEMBER3_PATH else '✗ NOT FOUND'} → {MEMBER3_PATH if MEMBER3_PATH else 'Using auto-detect'}")

if not MEMBER1_PATH and not MANUAL_MEMBER1_PATH:
    print(f"\n  Tried: {AUTO_MEMBER1_CANDIDATES}")
if not MEMBER2_PATH and not MANUAL_MEMBER2_PATH:
    print(f"\n  Tried: {AUTO_MEMBER2_CANDIDATES}")
if not MEMBER3_PATH and not MANUAL_MEMBER3_PATH:
    print(f"\n  Tried: {AUTO_MEMBER3_CANDIDATES}")
print("="*60)

# Check if we have at least one model
if not any([MEMBER1_PATH, MEMBER2_PATH, MEMBER3_PATH]):
    print("\n❌ CRITICAL: No model weights found!")
    print("Please upload your .pth files to Kaggle and ensure they're in one of these locations:")
    print("  1. /kaggle/input/member1-weights/ (create a dataset)")
    print("  2. /kaggle/input/{your-dataset-name}/")
    print("  3. /kaggle/working/ (upload via notebook)")
    print("\nOr manually set MEMBER1_PATH, MEMBER2_PATH, MEMBER3_PATH in the code above.")
    raise RuntimeError("No model weights found — see messages above.")

os.makedirs(CONFIG['output_dir'], exist_ok=True)
for sub in ['masks', 'masks_color', 'comparisons']:
    os.makedirs(f"{CONFIG['output_dir']}/predictions/{sub}", exist_ok=True)

h, w         = CONFIG['img_h'], CONFIG['img_w']
tokenH       = h // 14   # 18
tokenW       = w // 14   # 33

print(f"\nImage: {h}×{w} | Token grid: {tokenH}×{tokenW}")
print("Configuration ready ✓")


# ============================================================
# CELL 3 — CLASS MAPPING & COLORS
# ============================================================
# 16-bit mask raw values → class IDs 0-9
# IDENTICAL across all 3 members — must not change
VALUE_MAP = {
    0    : 0,   # Background
    100  : 1,   # Trees
    200  : 2,   # Lush Bushes
    300  : 3,   # Dry Grass
    500  : 4,   # Dry Bushes
    550  : 5,   # Ground Clutter
    700  : 6,   # Logs        ← rarest: 0.07% of pixels
    800  : 7,   # Rocks
    7100 : 8,   # Landscape
    10000: 9,   # Sky
}

CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

# RGB palette for visualization (matches Person 3's palette exactly)
COLOR_PALETTE = np.array([
    [0,   0,   0  ],  # 0 Background    — black
    [34,  139, 34 ],  # 1 Trees         — forest green
    [0,   255, 0  ],  # 2 Lush Bushes   — lime
    [210, 180, 140],  # 3 Dry Grass     — tan
    [139, 90,  43 ],  # 4 Dry Bushes    — brown
    [128, 128, 0  ],  # 5 Ground Clutter— olive
    [139, 69,  19 ],  # 6 Logs          — saddle brown
    [128, 128, 128],  # 7 Rocks         — gray
    [160, 82,  45 ],  # 8 Landscape     — sienna
    [135, 206, 235],  # 9 Sky           — sky blue
], dtype=np.uint8)

RARE_CLASSES = [5, 6]  # Ground Clutter, Logs

print("Class mapping defined ✓")


# ============================================================
# CELL 4 — SHARED MODEL ARCHITECTURE
# ============================================================
# SegmentationHeadConvNeXt — IDENTICAL across all 3 members.
# Member 1 notebook cell 7, Person 3 notebook cell 6, Member 2 cell 10
# all define the EXACT same architecture.
# If you change anything here, weights will fail to load.

class SegmentationHeadConvNeXt(nn.Module):
    """
    ConvNeXt-style segmentation head on DINOv2 patch tokens.
    Input  : [B, N, C]  where N = tokenH*tokenW, C = 384 (vits14 embed dim)
    Output : [B, n_classes, tokenH, tokenW]  (caller upsamples to full res)
    """
    def __init__(self, in_channels, out_channels, tokenW, tokenH, dropout=0.1):
        super().__init__()
        self.tokenH = tokenH
        self.tokenW = tokenW
        # stem: project from embed_dim → 256 with large kernel
        self.stem   = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=7, padding=3),
            nn.GELU(),
        )
        # block1: depthwise-sep convolution (ConvNeXt style)
        self.block1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.GELU(),
        )
        # block2: reduce to 128 channels
        self.block2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.dropout    = nn.Dropout2d(p=dropout)
        self.classifier = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        """x: [B, N, C]  →  [B, n_classes, tokenH, tokenW]"""
        B, N, C = x.shape
        # Reshape patch tokens → 2-D spatial grid
        x = x.reshape(B, self.tokenH, self.tokenW, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.dropout(x)
        return self.classifier(x)


# ============================================================
# CELL 5 — LOAD DINOV2 BACKBONE  (shared, frozen, eval only)
# ============================================================
print("\nLoading DINOv2-vits14 backbone (shared across all 3 heads)...")
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14',
                          pretrained=True, verbose=False)
backbone = backbone.to(device).eval()
for p in backbone.parameters():
    p.requires_grad = False

# Probe embedding dimension + verify token grid
with torch.no_grad():
    _probe = torch.zeros(1, 3, h, w, device=device)
    _feat  = backbone.forward_features(_probe)['x_norm_patchtokens']
    N_EMB  = _feat.shape[2]   # 384 for vits14
    _got_N = _feat.shape[1]
    _exp_N = tokenH * tokenW
    assert _got_N == _exp_N, \
        f"Token grid mismatch: expected {_exp_N} ({tokenH}×{tokenW}), got {_got_N}"

print(f"✓ Backbone loaded | Embed dim: {N_EMB} | Tokens: {tokenH}×{tokenW} = {_exp_N}")


# ============================================================
# CELL 6 — LOAD INDIVIDUAL HEADS
# ============================================================
# All 3 members save plain state_dicts (torch.save(classifier.state_dict(), path))
# so the loading logic is uniform.

def _try_unzip(path: str) -> str:
    """If path has .zip extension, extract it and return the first .pth inside."""
    if path.endswith('.zip') and os.path.isfile(path):
        out_dir = path[:-4] + '_extracted'
        os.makedirs(out_dir, exist_ok=True)
        with zipfile.ZipFile(path, 'r') as zf:
            zf.extractall(out_dir)
        pths = list(Path(out_dir).rglob('*.pth'))
        if not pths:
            raise FileNotFoundError(f"No .pth found after extracting {path}")
        return str(pths[0])
    return path


def load_head(ckpt_path: str, member_name: str) -> SegmentationHeadConvNeXt | None:
    """
    Load a SegmentationHeadConvNeXt from a checkpoint file.

    Handles three formats:
      A) plain state_dict          → torch.save(model.state_dict(), path)
         Used by: Member 1 Phase 2, Member 2, Member 3
      B) wrapped dict with key     → {'classifier_state': state_dict, ...}
         Used by: Member 1 Phase 1 checkpoints (phase1_best.pth)
      C) nn.Module directly        → rare, handled as fallback

    Returns head in eval() mode, or None if loading fails.
    """
    if ckpt_path is None:
        print(f"  ⚠  {member_name}: No path provided (skipped)")
        return None
    
    resolved = _try_unzip(ckpt_path)

    if not os.path.exists(resolved):
        print(f"  ⚠  {member_name}: file not found — {ckpt_path}")
        return None

    head = SegmentationHeadConvNeXt(N_EMB, CONFIG['n_classes'], tokenW, tokenH).to(device)

    try:
        obj = torch.load(resolved, map_location=device, weights_only=False)
    except Exception as e:
        print(f"  ⚠  {member_name}: torch.load failed — {e}")
        return None

    # ── Identify the actual state dict inside the checkpoint ──
    if isinstance(obj, dict):
        if 'classifier_state' in obj:
            # Member 1 Phase 1 checkpoint format
            state = obj['classifier_state']
        elif 'state_dict' in obj:
            state = obj['state_dict']
        elif 'model' in obj:
            state = obj['model']
        else:
            # Plain state dict (Member 2, Member 3, Member 1 Phase 2)
            state = obj
    elif isinstance(obj, nn.Module):
        state = obj.state_dict()
    else:
        print(f"  ⚠  {member_name}: unrecognised checkpoint type {type(obj)}")
        return None

    # ── Load with strict=True first, fall back to strict=False ──
    try:
        missing, unexpected = head.load_state_dict(state, strict=True)
        pct = 100.0
        print(f"  ✓ {member_name}: strict load OK ({len(state)} tensors)")
    except RuntimeError:
        missing, unexpected = head.load_state_dict(state, strict=False)
        loaded  = len(state) - len(missing)
        pct     = loaded / max(len(state), 1) * 100
        print(f"  ~ {member_name}: partial ({pct:.0f}%) | "
              f"missing={len(missing)}, unexpected={len(unexpected)}")
        if pct < 60:
            print(f"    ⚠  Only {pct:.0f}% matched — verify this is the right checkpoint!")

    head.eval()
    return head


print("\n" + "="*60)
print("Loading individual model heads...")
print("="*60)

m1 = load_head(CONFIG['member1_path'], "Member 1 (2-phase fine-tune)")
m2 = load_head(CONFIG['member2_path'], "Member 2 (augmentation)")
m3 = load_head(CONFIG['member3_path'], "Member 3 (OneCycleLR hyper)")

# Build list of (name, head, weight) — skip any that failed to load
_raw = [
    ("Member 1", m1, CONFIG['ensemble_weights'][0]),
    ("Member 2", m2, CONFIG['ensemble_weights'][1]),
    ("Member 3", m3, CONFIG['ensemble_weights'][2]),
]
loaded_heads = [(n, h_m, wt) for n, h_m, wt in _raw if h_m is not None]

print(f"\n{'='*40}")
print(f"Active heads: {len(loaded_heads)}/3")
for n, _, wt in loaded_heads:
    print(f"  • {n:30s} weight={wt:.2f}")
print(f"{'='*40}")

if not loaded_heads:
    raise RuntimeError("❌ No heads loaded — cannot run ensemble. Check paths above.")


# ============================================================
# CELL 7 — DATASET
# ============================================================

def convert_mask_np(mask_pil: Image.Image) -> np.ndarray:
    """16-bit PIL mask → uint8 class IDs 0-9."""
    arr = np.array(mask_pil)
    out = np.zeros_like(arr, dtype=np.uint8)
    for raw_val, class_id in VALUE_MAP.items():
        out[arr == raw_val] = class_id
    return out


# Use torchvision transforms (same as Person 1 + Person 3 notebooks)
img_transform = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
mask_transform = transforms.Compose([
    transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])


class DesertValDataset(Dataset):
    """Validation set — returns (img_tensor, mask_tensor, filename)."""
    def __init__(self, data_dir):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.mask_dir  = os.path.join(data_dir, 'Segmentation')
        self.filenames = sorted(os.listdir(self.image_dir))

    def __len__(self): return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img   = img_transform(
            Image.open(os.path.join(self.image_dir, fname)).convert('RGB'))
        mask_pil  = Image.open(os.path.join(self.mask_dir, fname))
        mask_cls  = Image.fromarray(convert_mask_np(mask_pil))
        # mask_transform returns [1,H,W] float → multiply ×255 → long → squeeze
        mask_t    = (mask_transform(mask_cls) * 255).long().squeeze(0)  # [H, W]
        return img, mask_t, fname


class DesertTestDataset(Dataset):
    """Test set (no masks) — returns (img_tensor, filename)."""
    def __init__(self, data_dir):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.filenames = sorted(os.listdir(self.image_dir))

    def __len__(self): return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img   = img_transform(
            Image.open(os.path.join(self.image_dir, fname)).convert('RGB'))
        return img, fname


val_dataset  = DesertValDataset(CONFIG['val_dir'])
test_dataset = DesertTestDataset(CONFIG['test_dir'])

val_loader  = DataLoader(val_dataset,  batch_size=CONFIG['batch_size'], shuffle=False,
                         num_workers=CONFIG['num_workers'], pin_memory=True,
                         persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
                         num_workers=CONFIG['num_workers'], pin_memory=True,
                         persistent_workers=True)

print(f"\nDatasets | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# Sanity check
_imgs, _masks, _ = next(iter(val_loader))
assert _imgs.shape == torch.Size([CONFIG['batch_size'], 3, h, w]), \
    f"Image shape mismatch: {_imgs.shape}"
assert _masks.max() <= 9, f"Mask out of range: max={_masks.max()}"
assert _masks.dtype == torch.int64, f"Mask dtype should be int64, got {_masks.dtype}"
print(f"Batch check: imgs={_imgs.shape} masks={_masks.shape} dtype={_masks.dtype}")
print("Dataset sanity check PASSED ✓")


# ============================================================
# CELL 8 — INFERENCE UTILITIES
# ============================================================

@torch.no_grad()
def _forward_one(head: SegmentationHeadConvNeXt,
                 imgs: torch.Tensor) -> torch.Tensor:
    """
    One forward pass: backbone → head → upsample → softmax.
    Returns probability map [B, n_classes, H, W].
    imgs must already be on CPU or GPU (moved inside here).
    """
    imgs_gpu = imgs.to(device)
    B, _, H, W = imgs_gpu.shape
    feats    = backbone.forward_features(imgs_gpu)['x_norm_patchtokens']  # [B,N,C]
    logits   = head(feats)                                                  # [B,C,tH,tW]
    logits   = F.interpolate(logits, size=(H, W), mode='bilinear',
                             align_corners=False)                           # [B,C,H,W]
    return F.softmax(logits, dim=1)                                         # [B,C,H,W]


@torch.no_grad()
def _tta_two_variant(head: SegmentationHeadConvNeXt,
                     imgs: torch.Tensor) -> torch.Tensor:
    """
    2-variant TTA: original + horizontal flip.
    Scale variants are DISABLED because changing image size changes the token
    grid (H//14 × W//14) which this fixed-grid classifier cannot handle.
    Confirmed by Person 3's notebook output.
    Returns averaged probs [B, n_classes, H, W].
    """
    # variant 1: original
    p1 = _forward_one(head, imgs)
    # variant 2: h-flip image, then flip prediction back
    flipped = torch.flip(imgs, dims=[-1])
    p2      = torch.flip(_forward_one(head, flipped), dims=[-1])
    return (p1 + p2) / 2.0


@torch.no_grad()
def ensemble_predict(imgs: torch.Tensor, use_tta: bool) -> torch.Tensor:
    """
    Run all active heads on a batch, weight-average softmax probs, return argmax.
    Args:
        imgs    : [B, 3, H, W]  normalized image batch (on CPU is fine)
        use_tta : if True, apply 2-variant TTA per head
    Returns:
        preds   : [B, H, W]  long — class IDs 0-9
    """
    B, _, H, W = imgs.shape
    total_weight  = sum(wt for _, _, wt in loaded_heads)
    combined      = torch.zeros(B, CONFIG['n_classes'], H, W, device=device)

    for _, head, wt in loaded_heads:
        if use_tta:
            probs = _tta_two_variant(head, imgs)
        else:
            probs = _forward_one(head, imgs)
        combined += wt * probs

    combined /= total_weight
    return torch.argmax(combined, dim=1)   # [B, H, W]


# ============================================================
# CELL 9 — mIoU METRIC
# ============================================================

def compute_miou(preds: torch.Tensor, targets: torch.Tensor,
                 n_classes: int = 10):
    """
    preds   : [B, H, W] long  (already argmax, not logits)
    targets : [B, H, W] long
    Returns : (mean_iou: float, per_class: list[float])
    """
    preds_f   = preds.view(-1).cpu()
    targets_f = targets.view(-1).cpu()
    iou_list  = []
    for c in range(n_classes):
        pc = preds_f == c; tc = targets_f == c
        inter = (pc & tc).sum().float()
        union = (pc | tc).sum().float()
        iou_list.append(float('nan') if union == 0 else (inter / union).item())
    return float(np.nanmean(iou_list)), iou_list


def print_iou_table(mean_iou: float, iou_list: list, title: str = ""):
    print(f"\n{'='*55}")
    if title: print(f"  {title}")
    print(f"  Mean IoU : {mean_iou:.4f}")
    print(f"{'='*55}")
    for i, (name, iou) in enumerate(zip(CLASS_NAMES, iou_list)):
        s   = f"{iou:.4f}" if not np.isnan(iou) else "  N/A"
        bar = '█' * int((iou if not np.isnan(iou) else 0) * 20)
        rare = " ← RARE" if i in RARE_CLASSES else ""
        print(f"  [{i}] {name:<16}: {s}  {bar}{rare}")


# ============================================================
# CELL 10 — VALIDATION EVALUATION
# ============================================================
print("\n" + "="*60)
print("VALIDATION EVALUATION")
print(f"Heads: {len(loaded_heads)}/3  |  TTA: {'ON (2-var)' if CONFIG['use_tta'] else 'OFF'}")
print(f"Weights: {CONFIG['ensemble_weights']}")
print("="*60)

# Per-member tracking
member_ious       = {n: [] for n, _, _ in loaded_heads}
member_class_ious = {n: [] for n, _, _ in loaded_heads}
ens_ious          = []
ens_class_ious    = []

with torch.no_grad():
    for imgs, masks, _ in tqdm(val_loader, desc="Val eval", unit="batch"):
        # Ensemble prediction
        preds_ens = ensemble_predict(imgs, CONFIG['use_tta'])  # [B, H, W]
        iou, iou_list = compute_miou(preds_ens, masks)
        ens_ious.append(iou)
        ens_class_ious.append(iou_list)

        # Per-member (no TTA for speed in this loop)
        for name, head, _ in loaded_heads:
            probs_m   = _forward_one(head, imgs)     # [B, C, H, W]
            preds_m   = torch.argmax(probs_m, dim=1) # [B, H, W]
            iou_m, iou_list_m = compute_miou(preds_m, masks)
            member_ious[name].append(iou_m)
            member_class_ious[name].append(iou_list_m)

# ── Individual member report ──────────────────────────────────
print("\n── Individual member mIoU (no TTA):")
for name, _, wt in loaded_heads:
    m = float(np.nanmean(member_ious[name]))
    print(f"  {name:30s}: {m:.4f}  (weight={wt:.2f})")

# ── Ensemble report ───────────────────────────────────────────
val_miou      = float(np.nanmean(ens_ious))
val_class_avg = np.nanmean(ens_class_ious, axis=0).tolist()
print_iou_table(val_miou, val_class_avg,
                title=f"ENSEMBLE Val mIoU (TTA={'ON' if CONFIG['use_tta'] else 'OFF'})")

# ── Save per-class bar chart ──────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
bar_vals = [0 if np.isnan(v) else v for v in val_class_avg]
colors   = ['red' if v < 0.3 else 'orange' if v < 0.5 else 'steelblue' for v in bar_vals]
ax.bar(CLASS_NAMES, bar_vals, color=colors, edgecolor='black', alpha=0.85)
ax.axhline(y=val_miou, color='red', linestyle='--', linewidth=2,
           label=f'Ensemble: {val_miou:.4f}')
palette = ['green', 'purple', 'darkorange']
for i, (name, _, _) in enumerate(loaded_heads):
    m = float(np.nanmean(member_ious[name]))
    ax.axhline(y=m, color=palette[i % len(palette)], linestyle=':', linewidth=1.5,
               label=f'{name}: {m:.4f}')
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
ax.set_title('Per-Class IoU — Ensemble vs Individual Members')
ax.set_ylabel('IoU'); ax.set_ylim(0, 1); ax.legend(); ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
chart_path = f"{CONFIG['output_dir']}/val_per_class_iou.png"
plt.savefig(chart_path, dpi=150); plt.close()
print(f"\n✓ Bar chart saved: {chart_path}")


# ============================================================
# CELL 11 — TEST SET INFERENCE
# ============================================================
print("\n" + "="*60)
print("TEST SET INFERENCE")
print(f"Images: {len(test_dataset)} | TTA: {'ON (2-var)' if CONFIG['use_tta'] else 'OFF'}")
print("="*60)

MEAN_NP = np.array([0.485, 0.456, 0.406])
STD_NP  = np.array([0.229, 0.224, 0.225])

pred_dir  = f"{CONFIG['output_dir']}/predictions/masks"
color_dir = f"{CONFIG['output_dir']}/predictions/masks_color"
comp_dir  = f"{CONFIG['output_dir']}/predictions/comparisons"

saved_count = 0
comp_count  = 0

with torch.no_grad():
    pbar = tqdm(test_loader, desc="Test inference", unit="batch")
    for imgs, fnames in pbar:
        preds = ensemble_predict(imgs, CONFIG['use_tta'])  # [B, H, W]

        for i in range(imgs.shape[0]):
            fname     = fnames[i]
            base      = os.path.splitext(fname)[0]
            pred_mask = preds[i].cpu().numpy().astype(np.uint8)   # [H, W]

            # ── Raw mask (class IDs as uint8 PNG) ──────────
            Image.fromarray(pred_mask).save(
                os.path.join(pred_dir, f"{base}_pred.png"))

            # ── Colorized mask ──────────────────────────────
            color_mask = COLOR_PALETTE[pred_mask]  # [H, W, 3] fancy index
            cv2.imwrite(
                os.path.join(color_dir, f"{base}_color.png"),
                cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))

            # ── Side-by-side comparison (first 10) ─────────
            if comp_count < 10:
                img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
                img_np = np.clip(img_np * STD_NP + MEAN_NP, 0, 1)
                img_np = (img_np * 255).astype(np.uint8)
                comp   = np.hstack([img_np, color_mask])
                cv2.imwrite(
                    os.path.join(comp_dir, f"sample_{comp_count:03d}_{base}.png"),
                    cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
                comp_count += 1

            saved_count += 1
        pbar.set_postfix(saved=saved_count)

print(f"\n✓ Inference done!")
print(f"  Saved : {saved_count} | Expected: {len(test_dataset)}")
assert saved_count == len(test_dataset), \
    f"❌ Count mismatch! {saved_count} vs {len(test_dataset)}"
print(f"  Count : ✓")


# ============================================================
# CELL 12 — FINAL SUMMARY
# ============================================================
best_individual = max(np.nanmean(member_ious[n]) for n, _, _ in loaded_heads)

print("\n" + "="*65)
print("✅  FINAL ENSEMBLE COMPLETE")
print("="*65)
print(f"\n  Active heads ({len(loaded_heads)}/3):")
for n, _, wt in loaded_heads:
    m = float(np.nanmean(member_ious[n]))
    print(f"    • {n:30s} weight={wt:.2f}  member_mIoU={m:.4f}")

print(f"\n  Ensemble mIoU  : {val_miou:.4f}  "
      f"({'↑ above best individual' if val_miou > best_individual else '↓ below best — check weights'})")
print(f"  TTA            : {'ON — 2 variants (orig + hflip)' if CONFIG['use_tta'] else 'OFF'}")
print(f"  Test images    : {saved_count}")
print(f"\n  Outputs:")
print(f"  {CONFIG['output_dir']}/predictions/")
print(f"  ├── masks/        ← {saved_count} raw uint8 masks (class IDs 0-9)")
print(f"  ├── masks_color/  ← {saved_count} colorized RGB masks")
print(f"  └── comparisons/  ← {comp_count} input|prediction panels")
print(f"\n  Per-class chart: val_per_class_iou.png")
print("="*65)
