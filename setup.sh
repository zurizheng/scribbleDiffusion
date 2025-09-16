#!/bin/bash

# ScribbleDiffusion Quick Start Script
# This script sets up the environment and runs a quick demo

echo "🎨 ScribbleDiffusion Quick Start"
echo "================================="

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip first."
    exit 1
fi

echo "✅ Python and pip found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Run setup script
echo "⚙️  Setting up environment..."
python scripts/download_models.py

python scripts/download_models.py

# Check device availability
echo "🔍 Checking device availability..."
python -c "
import torch
import sys
sys.path.append('src')
from utils.device_utils import get_device_info

try:
    info = get_device_info()
    print(f'🚀 Device: {info[\"device\"]}')
    print(f'📱 Device name: {info[\"device_name\"]}')
    print(f'⚡ CUDA available: {info[\"cuda_available\"]}')
    print(f'🍎 MPS available: {info[\"mps_available\"]}')
    print(f'💻 GPU available: {info[\"gpu_available\"]}')
except Exception as e:
    print(f'⚠️  Could not check device info: {e}')
    print('📝 Manual check:')
    print(f'   CUDA: {torch.cuda.is_available()}')
    if hasattr(torch.backends, \"mps\"):
        print(f'   MPS: {torch.backends.mps.is_available()}')
    else:
        print('   MPS: Not supported (PyTorch < 1.12)')
" 2>/dev/null || echo "⚠️  PyTorch not installed yet - run pip install -r requirements.txt"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Train model: python scripts/train.py --config configs/fast.yaml"
echo "2. Run inference: python scripts/fixed_inference.py"
echo "3. Run demo: python app.py"
echo "3. Open notebooks: jupyter lab notebooks/"
echo ""
echo "For custom datasets, see src/data/dataset.py"
echo "For model details, see src/models/"
echo ""
echo "Happy sketching! 🎨"
