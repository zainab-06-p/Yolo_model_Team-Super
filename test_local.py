# ============================================================
# LOCAL SINGLE-IMAGE TEST — CPU, no Kaggle needed
# Tests DINOv2 + SegmentationHeadConvNeXt (Member 2 weights)
# Compares predicted segmentation against reference image
# ============================================================

import os, sys, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torchvision.transforms as transforms

warnings.filterwarnings('ignore')

# ── Config ───────────────────────────────────────────────────
device = torch.device('cpu')
IMG_H, IMG_W = 252, 462      # DINOv2-vits14 multiples of 14
TOKEN_H = IMG_H // 14        # 18
TOKEN_W = IMG_W // 14        # 33
N_CLASSES = 10
N_EMB = 384                  # dinov2_vits14 embed dim

INPUT_IMAGE  = r"3.jpg"
GROUND_TRUTH = r"round 2\after\3 after.jpg"
WEIGHT_PATH  = r"model_augmented_best.pth"  # Member 2 (local)
OUTPUT_DIR   = r"test_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"PyTorch {torch.__version__} | Device: CPU")
print(f"Image size: {IMG_H}×{IMG_W}  |  Token grid: {TOKEN_H}×{TOKEN_W}")

# ── Class definitions ─────────────────────────────────────────
VALUE_MAP = {
    0:0, 100:1, 200:2, 300:3, 500:4,
    550:5, 700:6, 800:7, 7100:8, 10000:9
}
CLASS_NAMES = [
    'Background','Trees','Lush Bushes','Dry Grass','Dry Bushes',
    'Ground Clutter','Logs','Rocks','Landscape','Sky'
]
COLOR_PALETTE = np.array([
    [0,   0,   0  ],  # 0 Background
    [34,  139, 34 ],  # 1 Trees
    [0,   200, 0  ],  # 2 Lush Bushes
    [210, 180, 140],  # 3 Dry Grass
    [139, 90,  43 ],  # 4 Dry Bushes
    [128, 128, 0  ],  # 5 Ground Clutter
    [139, 69,  19 ],  # 6 Logs
    [128, 128, 128],  # 7 Rocks
    [160, 82,  45 ],  # 8 Landscape
    [135, 206, 235],  # 9 Sky
], dtype=np.uint8)

# ── Model definition ──────────────────────────────────────────
class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH, dropout=0.1):
        super().__init__()
        self.tokenH, self.tokenW = tokenH, tokenW
        self.stem   = nn.Sequential(nn.Conv2d(in_channels, 256, 7, padding=3), nn.GELU())
        self.block1 = nn.Sequential(
            nn.Conv2d(256, 256, 7, padding=3, groups=256), nn.GELU(),
            nn.Conv2d(256, 256, 1), nn.GELU(),
        )
        self.block2    = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.GELU())
        self.dropout   = nn.Dropout2d(p=dropout)
        self.classifier= nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.tokenH, self.tokenW, C).permute(0, 3, 1, 2)
        x = self.stem(x); x = self.block1(x); x = self.block2(x)
        return self.classifier(self.dropout(x))

# ── Step 1: Load backbone ─────────────────────────────────────
print("\n[1/5] Loading DINOv2-vits14 backbone (CPU)...")
t0 = time.time()
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14',
                          pretrained=True, verbose=False)
backbone = backbone.eval()
for p in backbone.parameters():
    p.requires_grad = False
print(f"      ✓ Backbone loaded in {time.time()-t0:.1f}s")

# ── Step 2: Load head ─────────────────────────────────────────
print(f"\n[2/5] Loading Member 2 head from: {WEIGHT_PATH}")
head = SegmentationHeadConvNeXt(N_EMB, N_CLASSES, TOKEN_W, TOKEN_H)
state = torch.load(WEIGHT_PATH, map_location='cpu', weights_only=False)
# Handle wrapped vs plain state dict
if isinstance(state, dict) and 'classifier_state' in state:
    state = state['classifier_state']
elif isinstance(state, dict) and any(k.startswith('stem') or k.startswith('block') or k.startswith('classifier') for k in state):
    pass  # plain state dict — use as is
missing, unexpected = head.load_state_dict(state, strict=True)
head.eval()
total_params = sum(p.numel() for p in head.parameters())
print(f"      ✓ Head loaded | Params: {total_params:,} | Missing: {len(missing)} | Unexpected: {len(unexpected)}")

# ── Step 3: Preprocess both images ───────────────────────────
print(f"\n[3/5] Preprocessing images...")

img_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

MEAN_NP = np.array([0.485,0.456,0.406])
STD_NP  = np.array([0.229,0.224,0.225])

# Input image (edited — has cyan road)
img_pil   = Image.open(INPUT_IMAGE).convert('RGB')
orig_w, orig_h = img_pil.size
img_tensor = img_transform(img_pil).unsqueeze(0)   # [1,3,H,W]
print(f"      Input image  : {INPUT_IMAGE}  original={orig_w}×{orig_h}  → resized to {IMG_W}×{IMG_H}")

# Reference image (clean — the "after" version to compare against)
ref_pil   = Image.open(GROUND_TRUTH).convert('RGB')
ref_w, ref_h = ref_pil.size
ref_tensor = img_transform(ref_pil).unsqueeze(0)
print(f"      Reference img: {GROUND_TRUTH}  original={ref_w}×{ref_h}  → resized to {IMG_W}×{IMG_H}")

# For display — denormalize to uint8
def tensor_to_rgb(t):
    arr = t.squeeze(0).permute(1,2,0).numpy()
    return np.clip(arr * STD_NP + MEAN_NP, 0, 1)

input_rgb = tensor_to_rgb(img_tensor)    # [H,W,3] float 0-1
ref_rgb   = tensor_to_rgb(ref_tensor)

# ── Step 4: Run inference ─────────────────────────────────────
print(f"\n[4/5] Running inference (CPU — this takes ~30-60s)...")

def predict_single(img_t):
    """img_t: [1,3,H,W] → pred_mask [H,W] uint8, probs [C,H,W]"""
    t = time.time()
    with torch.no_grad():
        feats   = backbone.forward_features(img_t)['x_norm_patchtokens']  # [1,N,384]
        logits  = head(feats)                                               # [1,10,18,33]
        logits  = F.interpolate(logits, size=(IMG_H,IMG_W),
                                mode='bilinear', align_corners=False)       # [1,10,252,462]
        probs   = F.softmax(logits, dim=1)                                  # [1,10,252,462]
        pred    = torch.argmax(probs, dim=1).squeeze(0)                    # [252,462]
    elapsed = time.time() - t
    return pred.numpy().astype(np.uint8), probs.squeeze(0).numpy(), elapsed

# TTA (2-variant) for edited image
print("      → Pass 1: Edited image (original orientation)")
pred_input_orig, probs_orig, t1 = predict_single(img_tensor)

print("      → Pass 2: Edited image (h-flip TTA)")
img_flip = torch.flip(img_tensor, dims=[-1])
pred_flip_raw, probs_flip, t2 = predict_single(img_flip)
probs_flip_back = probs_flip[:, :, ::-1].copy()  # flip back
probs_tta = (probs_orig + probs_flip_back) / 2.0
pred_input = np.argmax(probs_tta, axis=0).astype(np.uint8)

print(f"      → Edited image TTA done ({t1+t2:.1f}s)")

print("      → Pass 3: Reference / 'clean' image")
pred_ref, probs_ref, t3 = predict_single(ref_tensor)
print(f"      → Reference image done ({t3:.1f}s)")

# ── Step 5: Compare & save results ───────────────────────────
print(f"\n[5/5] Computing comparison metrics & saving visuals...")

# Colourize predictions
color_input = COLOR_PALETTE[pred_input]   # [H,W,3]
color_ref   = COLOR_PALETTE[pred_ref]

# Per-class pixel distribution
def class_distribution(mask):
    dist = {}
    total = mask.size
    for i, name in enumerate(CLASS_NAMES):
        pct = (mask == i).sum() / total * 100
        dist[name] = pct
    return dist

dist_input = class_distribution(pred_input)
dist_ref   = class_distribution(pred_ref)

# Pixel-level agreement between the two predictions
agreement = (pred_input == pred_ref).mean() * 100
diff_mask  = (pred_input != pred_ref).astype(np.uint8) * 255  # white = different

# ── Print class-level comparison table ───────────────────────
print("\n" + "="*72)
print(f"  {'Class':<16} {'Edited %':>9} {'Reference %':>11} {'Diff':>8}  Agreement")
print("="*72)
for name in CLASS_NAMES:
    ei  = dist_input.get(name, 0)
    ri  = dist_ref.get(name, 0)
    d   = ei - ri
    sign = '+' if d >= 0 else ''
    bar = '█' * int(min(ei, ri) / 2)
    print(f"  {name:<16} {ei:>8.2f}% {ri:>10.2f}% {sign}{d:>7.2f}%  {bar}")
print("="*72)
print(f"\n  Overall pixel agreement (edited vs ref prediction): {agreement:.2f}%")
print(f"  Pixels that differ:  {100-agreement:.2f}%")

# ── Highlight diff pixels on the input ─────────────────────
# Red overlay where predictions diverge
diff_overlay = (input_rgb * 255).astype(np.uint8).copy()
diff_where   = (pred_input != pred_ref)
diff_overlay[diff_where] = [220, 30, 30]   # red = where predictions differ

# ── Save full comparison figure ──────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(20, 14))
fig.patch.set_facecolor('#0d1117')
titles_color = 'white'

title_style = dict(fontsize=11, color=titles_color, fontweight='bold', pad=6)

# Row 0: original images
axes[0,0].imshow(input_rgb);  axes[0,0].set_title("Input (Edited)", **title_style)
axes[0,1].imshow(ref_rgb);    axes[0,1].set_title("Reference (Clean)", **title_style)
axes[0,2].imshow(np.abs((input_rgb - ref_rgb) * 3).clip(0,1))
axes[0,2].set_title("Image Diff ×3 (Amplified)", **title_style)

# Row 1: segmentation results
axes[1,0].imshow(color_input)
axes[1,0].set_title(f"Predicted Mask — Edited", **title_style)
axes[1,1].imshow(color_ref)
axes[1,1].set_title(f"Predicted Mask — Reference", **title_style)
axes[1,2].imshow(diff_overlay)
axes[1,2].set_title(f"Prediction Diff (red = changed)  Agreement: {agreement:.1f}%", **title_style)

# Row 2: class distributions
cls_names_short = [n[:8] for n in CLASS_NAMES]
x = np.arange(N_CLASSES)
w = 0.38
vals_in  = [dist_input[n]  for n in CLASS_NAMES]
vals_ref = [dist_ref[n]    for n in CLASS_NAMES]
bar_colors = [f"#{r:02x}{g:02x}{b:02x}" for r,g,b in COLOR_PALETTE]
bar_colors[0] = '#555555'  # make background grey so it's visible

axes[2,0].bar(x - w/2, vals_in,  w, label='Edited',    alpha=0.85, color=bar_colors)
axes[2,0].bar(x + w/2, vals_ref, w, label='Reference', alpha=0.55, color=bar_colors, edgecolor='white')
axes[2,0].set_xticks(x); axes[2,0].set_xticklabels(cls_names_short, rotation=45, ha='right', color='white')
axes[2,0].set_title("Class Distribution Comparison (%)", **title_style)
axes[2,0].legend(facecolor='#1a1a2e', labelcolor='white')
axes[2,0].set_facecolor('#1a1a2e'); axes[2,0].tick_params(colors='white')
axes[2,0].yaxis.label.set_color('white'); axes[2,0].spines[:].set_color('#333')

# TTA confidence map — max prob across classes (how confident the model is)
conf_tta = probs_tta.max(axis=0)  # [H,W] 0-1
axes[2,1].imshow(conf_tta, cmap='RdYlGn', vmin=0.5, vmax=1.0)
axes[2,1].set_title("Model Confidence (TTA, Edited) — green=certain", **title_style)
plt.colorbar(axes[2,1].images[0], ax=axes[2,1], fraction=0.046)

# Class legend
handles = [mpatches.Patch(facecolor=bar_colors[i], label=f"[{i}] {CLASS_NAMES[i]}")
           for i in range(N_CLASSES)]
axes[2,2].legend(handles=handles, loc='center', fontsize=9,
                 facecolor='#1a1a2e', labelcolor='white',
                 framealpha=0.9, ncol=2)
axes[2,2].set_facecolor('#0d1117')
axes[2,2].set_title("Class Legend", **title_style)
axes[2,2].axis('off')

for ax in axes.flatten():
    ax.axis('off') if ax.has_data() else None
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

plt.suptitle(
    f"Ensemble Test — Member 2 (model_augmented_best.pth)\n"
    f"Overall Prediction Agreement (Edited vs Reference): {agreement:.2f}%",
    fontsize=14, color='white', fontweight='bold', y=1.01
)
plt.tight_layout(pad=0.8)

out_path = os.path.join(OUTPUT_DIR, "comparison_result.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()

# ── Save individual outputs ───────────────────────────────────
cv2.imwrite(os.path.join(OUTPUT_DIR, "pred_edited_color.png"),
            cv2.cvtColor(color_input, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(OUTPUT_DIR, "pred_ref_color.png"),
            cv2.cvtColor(color_ref, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(OUTPUT_DIR, "pred_edited_raw.png"), pred_input)
cv2.imwrite(os.path.join(OUTPUT_DIR, "pred_ref_raw.png"),    pred_ref)
cv2.imwrite(os.path.join(OUTPUT_DIR, "diff_mask.png"),       diff_mask)

# ── Final summary ─────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  ✅  LOCAL TEST COMPLETE")
print(f"{'='*55}")
print(f"  Model      : Member 2 — model_augmented_best.pth")
print(f"  Backbone   : dinov2_vits14 (CPU)")
print(f"  Input img  : {INPUT_IMAGE}")
print(f"  Ref img    : {GROUND_TRUTH}")
print(f"  TTA        : ON (2-variant: orig + hflip)")
print(f"\n  Agreement  : {agreement:.2f}%  ({'good' if agreement>75 else 'moderate' if agreement>55 else 'low — check model'} matching)")
print(f"\n  Top-3 diverging classes (edited vs ref):")
diffs = {n: abs(dist_input[n]-dist_ref[n]) for n in CLASS_NAMES}
for i, (nm, delta) in enumerate(sorted(diffs.items(), key=lambda x: -x[1])[:3]):
    print(f"    {i+1}. {nm:<16}: Δ{delta:.2f}%  "
          f"(edited={dist_input[nm]:.2f}%, ref={dist_ref[nm]:.2f}%)")

print(f"\n  Outputs saved to: {os.path.abspath(OUTPUT_DIR)}/")
print(f"    ├── comparison_result.png   ← full 3×3 panel")
print(f"    ├── pred_edited_color.png   ← colorized mask for edited input")
print(f"    ├── pred_ref_color.png      ← colorized mask for reference")
print(f"    ├── pred_edited_raw.png     ← raw class-ID mask (edited)")
print(f"    └── diff_mask.png           ← binary diff map")
print(f"{'='*55}")
