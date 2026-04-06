# Kaggle Notebook Instructions

## Setup

1. **Upload Dataset**: Add your `yolo-training-data` dataset to the notebook
   - Should contain: train/, val/, test/ folders with images

2. **Upload Models**: Add the 3 model weights
   - Option A: Upload to `/kaggle/working/` (drag & drop in notebook)
   - Option B: Create a dataset and add as input

3. **Update Paths**: Edit the CONFIG section in `ensemble_kaggle.py`:
   ```python
   MEMBER1_PATH = '/kaggle/working/model_finetuned_best.pth.zip'
   MEMBER2_PATH = '/kaggle/working/model_augmented_best.pth'
   MEMBER3_PATH = '/kaggle/working/model_best.pth.zip'
   DATASET_PATH = '/kaggle/input/yolo-training-data'
   ```

4. **Run**: Execute all cells

## Expected Output

- Validation mIoU for each model + ensemble
- Test predictions saved to `/kaggle/working/predictions/`
  - `masks/` - Raw class ID masks
  - `color/` - Colorized masks

## Troubleshooting

**Models not found?**
- Check the exact paths with: `!ls -R /kaggle/input/`
- Update MEMBER*_PATH variables accordingly

**Dataset not found?**
- Verify your dataset name matches DATASET_PATH
- Check folder structure with: `!find /kaggle/input -type d | head -20`

**CUDA OOM?**
- Reduce batch processing or use CPU
- The script processes images one at a time by default
