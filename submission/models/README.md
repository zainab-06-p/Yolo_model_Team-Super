# Ensemble Models Package

This folder contains all 3 model weights for the DINOv2 Offroad Segmentation Ensemble.

## Files Required

1. **model_finetuned_best.pth** (or .zip) - Member 1: Fine-tuned DINOv2
2. **model_augmented_best.pth** - Member 2: Augmentation + Class weights
3. **model_best.pth** (or .zip) - Member 3: Hyperparameter tuned

## How to Upload to Kaggle

### Method 1: Upload to /kaggle/working/ (Simplest)
1. In your Kaggle notebook, click "Add Input"
2. Upload these 3 files directly
3. They will appear in `/kaggle/working/`
4. Update paths in ensemble_kaggle.py if needed

### Method 2: Create a Kaggle Dataset (Recommended)
1. Go to Kaggle Datasets
2. Click "New Dataset"
3. Name: `offroad-ensemble-models`
4. Upload the 3 model files
5. In notebook: `/kaggle/input/offroad-ensemble-models/model_*.pth`

## Usage

See `ensemble_kaggle.py` for the complete inference code.
