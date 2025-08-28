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

# Check if CUDA is available
python -c "import torch; print('🚀 CUDA available:', torch.cuda.is_available())" 2>/dev/null || echo "⚠️  PyTorch not installed yet - run pip install -r requirements.txt"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Train model: python train.py --config configs/fast.yaml"
echo "2. Run demo: python app.py"
echo "3. Open notebooks: jupyter lab notebooks/"
echo ""
echo "For custom datasets, see src/data/dataset.py"
echo "For model details, see src/models/"
echo ""
echo "Happy sketching! 🎨"
