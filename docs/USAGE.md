# ScribbleDiffusion Usage Guide

## Project Structure (After Cleanup)

```
scribbleDiffusion/
├── app.py                     # Main web demo application
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── setup.sh                   # Setup script
├── start_training.sh          # Training launcher
├── src/                       # Source code modules
├── scripts/                   # All utility and main scripts
├── data/                      # Training data
├── configs/                   # Configuration files
├── checkpoints/               # Saved model checkpoints
├── scribble_diffusion_model/  # Exported model
└── docs/                      # Documentation
```

## Key Scripts in `/scripts/` Folder

### Training Scripts
- **`train.py`** - Main training script for ScribbleDiffusion
  ```bash
  python scripts/train.py --config configs/train_config.yaml
  ```

- **`train_cached.py`** - Optimized training with cached edge detection
  ```bash
  python scripts/train_cached.py --config configs/train_config.yaml
  ```

### Inference Scripts
- **`fixed_inference.py`** - ✅ **WORKING** inference pipeline that handles missing SketchTextCombiner
  ```bash
  python scripts/fixed_inference.py
  ```
  This is the main inference script that successfully generates images from sketches.

### Utilities
- **`convert_weights.py`** - Convert between different weight formats (linear ↔ conv)
  ```bash
  python scripts/convert_weights.py
  ```

- **`export_model.py`** - Export trained model to HuggingFace format
- **`upload_to_hf.py`** - Upload model to HuggingFace Hub
- **`download_from_hf.py`** - Download model from HuggingFace Hub

### Setup and Data
- **`download_models.py`** - Download base Stable Diffusion models
- **`setup_simple.py`** - Simple setup script
- **`analyze_dataset_diversity.py`** - Analyze training dataset

## Quick Start

1. **Setup Environment:**
   ```bash
   ./setup.sh
   ```

2. **Run Inference (CPU-optimized):**
   ```bash
   python scripts/fixed_inference.py
   ```

3. **Start Web Demo:**
   ```bash
   python app.py
   ```

4. **Train Model:**
   ```bash
   python scripts/train.py --config configs/train_config.yaml
   ```

## Important Notes

- **`scripts/fixed_inference.py`** is the main working inference script that resolves the SketchTextCombiner issue
- The project has been cleaned up - all debug and test files have been removed
- All major functionality is now organized in the `scripts/` folder
- `app.py` remains in root as the main demo entry point

## Hardware Requirements

- **Training:** GPU with 8GB+ VRAM recommended (CUDA or Apple Silicon M1/M2)
- **Inference:** Can run on CPU (4GB RAM), CUDA GPU, or Apple Silicon (MPS)
- **Demo:** Multi-device compatible with automatic device detection
- **Supported Devices:**
  - 🖥️ CUDA GPUs (NVIDIA)
  - 🍎 Apple Silicon (M1/M2/M3 with MPS)
  - 💻 CPU fallback (Intel/AMD)

## Device Detection

The project automatically detects and uses the best available device:
1. **CUDA** (NVIDIA GPU) - Highest priority
2. **MPS** (Apple Silicon GPU) - Second priority  
3. **CPU** - Fallback option

Test device support: `python scripts/test_device_support.py`

## Troubleshooting

If you encounter issues:
1. Check if weights need conversion: `python scripts/convert_weights.py`
2. Use CPU inference for memory constraints: Set `force_cpu=True` in scripts
3. Refer to the working `fixed_inference.py` for reference implementation