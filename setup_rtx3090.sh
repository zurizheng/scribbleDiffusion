#!/bin/bash
# RTX 3090 Setup Script for ScribbleDiffusion
# Run this script on your RunPod instance

set -e  # Exit on any error

echo "🚀 Setting up ScribbleDiffusion for RTX 3090 24GB VRAM"
echo "=" * 60

# Check if we're in the right directory
if [ ! -f "train.py" ]; then
    echo "❌ Error: train.py not found. Please run this script from the ScribbleDiffusion directory."
    exit 1
fi

# Check GPU
echo "📊 Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits

# Activate virtual environment
echo "🐍 Setting up Python environment..."
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

source .venv/bin/activate
echo "Virtual environment activated"

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support for RTX 3090
echo "🔥 Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other ML dependencies
echo "📦 Installing ML libraries..."
pip install accelerate transformers diffusers datasets

# Install additional dependencies
echo "📦 Installing additional dependencies..."
pip install -r requirements.txt

# Install development tools
echo "🛠️ Installing development tools..."
pip install tensorboard wandb opencv-python pillow numpy scipy matplotlib

# Verify PyTorch CUDA installation
echo "✅ Verifying PyTorch CUDA installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Test diffusers installation
echo "🧪 Testing diffusers installation..."
python -c "
import diffusers
print(f'Diffusers version: {diffusers.__version__}')
from diffusers import AutoencoderKL
print('✅ Diffusers working correctly')
"

# Create data directory if it doesn't exist
echo "📁 Setting up data directories..."
mkdir -p data/coco
mkdir -p outputs/rtx3090_training
mkdir -p logs/rtx3090
mkdir -p cache

# Check if COCO data exists
if [ ! -d "data/coco/train2017" ]; then
    echo "📊 COCO dataset not found. You can download it later or use the download script."
    echo "To download COCO dataset, run: bash data/coco/download_coco.sh"
else
    echo "✅ COCO dataset found"
fi

# Test model loading
echo "🧪 Testing model components..."
python -c "
import torch
from src.models.unet import SketchConditionedUNet
from src.models.hint_encoder import HintEncoder

print('Testing UNet initialization...')
unet = SketchConditionedUNet(
    in_channels=4,
    out_channels=4,
    model_channels=320,
    attention_resolutions=[1, 2, 4, 8],
    num_res_blocks=2,
    channel_mult=[1, 2, 4, 4],
    num_heads=8,
    use_spatial_transformer=True,
    transformer_depth=2,
    context_dim=768,
    use_checkpoint=False
)
print('✅ UNet initialized successfully')

print('Testing HintEncoder initialization...')
hint_encoder = HintEncoder(
    in_channels=1,
    hint_channels=[64, 128, 256, 512],
    injection_layers=[0, 1, 2, 3],
    injection_method='add',
    unet_channels=[320, 640, 1280, 1280]
)
print('✅ HintEncoder initialized successfully')

# Test GPU memory
if torch.cuda.is_available():
    device = torch.device('cuda')
    unet = unet.to(device)
    hint_encoder = hint_encoder.to(device)
    print(f'✅ Models loaded to GPU successfully')
    print(f'GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
    print(f'GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB')
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "🚀 Ready to start training!"
echo ""
echo "To start training with RTX 3090 config:"
echo "  python train.py --config configs/rtx3090.yaml"
echo ""
echo "To start training with standard config:"
echo "  python train.py --config configs/coco.yaml"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir logs/rtx3090"
echo ""
echo "💡 Don't forget to download COCO dataset if training on real data:"
echo "  bash data/coco/download_coco.sh"
echo ""
echo "Happy training! 🔥"
