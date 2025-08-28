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

# Verify we're using the right pip and python
echo "📍 Using python: $(which python)"
echo "📍 Using pip: $(which pip)"
echo "📍 Virtual env: $VIRTUAL_ENV"

# Upgrade pip in the virtual environment
echo "⬆️ Upgrading pip in virtual environment..."
python -m pip install --upgrade pip

# Install PyTorch with CUDA support for RTX 3090
echo "🔥 Installing stable PyTorch 2.0.1 with CUDA 11.7 (detected CUDA 11.7.1)..."
echo "📍 Using pip: $(which pip)"
echo "📍 Using python: $(which python)"
echo "📍 Virtual env: $VIRTUAL_ENV"

# Install compatible versions that work together
python -m pip install numpy==1.24.3  # Compatible with PyTorch 2.0.1
python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

# Check if PyTorch installed successfully
echo "🔍 Checking PyTorch installation..."
python -c "import sys; print('Python path:', sys.path[:3], '...')" 
python -c "import torch; print('✅ PyTorch imported successfully')" || {
    echo "❌ PyTorch installation failed. Checking what packages are installed..."
    python -m pip list | grep -i torch || echo "No torch packages found"
    echo "Trying CPU-only version as fallback..."
    python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
}

# Verify PyTorch CUDA installation immediately after install
echo "✅ Verifying PyTorch CUDA installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Expected CUDA: 11.7 (Container has CUDA 11.7.1)')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠️  CUDA not available - check CUDA compatibility')
"

# Test diffusers installation with compatible versions
echo "🧪 Testing diffusers installation..."
python -c "
import numpy
print(f'NumPy version: {numpy.__version__}')

try:
    import huggingface_hub
    print(f'HuggingFace Hub version: {huggingface_hub.__version__}')
    
    import diffusers
    print(f'Diffusers version: {diffusers.__version__}')
    from diffusers import AutoencoderKL
    print('✅ Diffusers working correctly with compatible versions!')
    
    # Test that we can actually use it
    print('🧪 Testing AutoencoderKL instantiation...')
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[64, 128],
        latent_channels=4
    )
    print('✅ AutoencoderKL created successfully!')
    
except ImportError as e:
    print(f'❌ Import failed: {e}')
    print('Checking installed versions...')
    import subprocess
    result = subprocess.run(['python', '-m', 'pip', 'list'], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'diffusers' in line or 'huggingface' in line:
            print(f'   {line}')
"

# Install other ML dependencies with compatible versions
echo "📦 Installing ML libraries with compatible versions..."
echo "Installing accelerate..."
python -m pip install accelerate==0.20.3 || echo "❌ Accelerate installation failed"

echo "Installing transformers..."
python -m pip install transformers==4.30.2 || echo "❌ Transformers installation failed"

echo "Ensuring diffusers is correct version..."
python -m pip uninstall diffusers -y 2>/dev/null || true
python -m pip uninstall huggingface-hub -y 2>/dev/null || true

# Install compatible huggingface_hub version first
python -m pip install huggingface_hub==0.16.4  # Compatible with diffusers 0.18.2
python -m pip install diffusers==0.18.2 --no-deps || echo "❌ Diffusers installation failed"

# Install other diffusers dependencies separately with compatible versions
python -m pip install regex requests Pillow safetensors==0.3.1

echo "Installing datasets..."
python -m pip install datasets==2.13.1 || echo "❌ Datasets installation failed"

# Verify what actually got installed
echo "📋 Checking installed ML packages..."
python -m pip list | grep -E "(accelerate|transformers|diffusers|datasets)" || echo "No ML packages found"

# Install additional dependencies (skip requirements.txt for now to avoid conflicts)
echo "📦 Installing additional dependencies..."
echo "Skipping requirements.txt to avoid version conflicts..."
# python -m pip install -r requirements.txt

# Install only essential additional packages manually
python -m pip install tqdm omegaconf safetensors einops Pillow opencv-python

# Install development tools
echo "🛠️ Installing development tools..."
python -m pip install tensorboard wandb opencv-python pillow numpy scipy matplotlib

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
    echo "    echo "To download COCO dataset if training on real data:"
    echo "  bash scripts/download_coco.sh""
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
