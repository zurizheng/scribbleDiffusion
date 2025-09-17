# RunPod Setup for ScribbleDiffusion Fruit Training

## Quick Start Guide

### 1. Download Your Dataset from Hugging Face
```bash
# Download your uploaded fruit dataset
huggingface-hub download zurizheng/my-fruit-dataset --repo-type=dataset --local-dir=my-fruit-dataset

# Verify dataset structure
ls my-fruit-dataset/
# Should show: Apple_Good/ Banana_Good/ Guava_Good/ Lime_Good/ Orange_Good/ Pomegranate_Good/
```

**Note**: The training config is already updated to use `my-fruit-dataset` as the data directory.

### 2. RunPod Instance Setup
Choose a pod with:
- **GPU**: RTX 4090, A100, or H100 (recommended)
- **VRAM**: At least 16GB for comfortable training
- **Disk**: 50GB+ for models and checkpoints
- **Template**: PyTorch or basic Ubuntu with CUDA

### 3. Environment Setup Commands
```bash
# Clone your repository
git clone https://github.com/zurizheng/scribbleDiffusion.git
cd scribbleDiffusion

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate safetensors
pip install opencv-python pillow numpy omegaconf
pip install tensorboard huggingface_hub

# Download your fruit dataset from Hugging Face
huggingface-hub download zurizheng/my-fruit-dataset --repo-type=dataset --local-dir=my-fruit-dataset

# Verify dataset structure
ls my-fruit-dataset/
# Should show: Apple_Good/ Banana_Good/ Guava_Good/ Lime_Good/ Orange_Good/ Pomegranate_Good/
```

### 4. Test Setup
```bash
# Verify dataset loading (should show 10,476 images after grayscale removal)
python3 test_fruit_setup.py
```

### 5. Start Training
```bash
# Start training with tensorboard logging
python3 scripts/train.py --config configs/fruit_training.yaml
```

### 6. Monitor Training
```bash
# In another terminal, start tensorboard
tensorboard --logdir outputs/fruit_model/logs --host 0.0.0.0 --port 6006
```
Then access via RunPod's port forwarding for port 6006.

## Training Configuration

Your `configs/fruit_training.yaml` is optimized for:
- **11,664 fruit images** across 6 categories
- **10,000 training steps** (~2-3 hours on RTX 4090)
- **256x256 resolution** for optimal quality/speed
- **Automatic text conditioning**: "a apple", "a banana", etc.
- **Sketch generation**: Automatic edge detection from your images

## Expected Results

After training, your model will:
✅ Generate different fruits based on text prompts
✅ Follow sketch inputs for shape guidance  
✅ Avoid kaleidoscope patterns from focused dataset
✅ Save complete model components for inference

## File Transfer Tips

**From local to RunPod:**
```bash
# Zip your dataset
tar -czf fruit_dataset.tar.gz data/fruits

# Upload via RunPod interface or:
scp fruit_dataset.tar.gz root@<runpod-ip>:/workspace/scribbleDiffusion/
```

**Download trained model:**
```bash
# From RunPod, compress your trained model
tar -czf fruit_model.tar.gz outputs/fruit_model

# Download via RunPod interface or scp back to local
```

## Cost Optimization

- **RTX 4090**: ~$0.50/hour - Good for 11k dataset
- **A100**: ~$1.50/hour - Faster training, overkill for this size
- **Spot instances**: 50% cheaper but can be interrupted

Estimated cost for complete training: **$2-5** on RTX 4090.