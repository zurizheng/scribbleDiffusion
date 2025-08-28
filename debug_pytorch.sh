#!/bin/bash
# Simple PyTorch debug installer for troubleshooting

echo "🔧 PyTorch Debug Installation"
echo "================================"

# Show environment info
echo "📍 Environment Info:"
echo "   Python: $(which python)"
echo "   Pip: $(which pip)"
echo "   Virtual env: $VIRTUAL_ENV"
echo "   Working dir: $(pwd)"

# Check pip works
echo ""
echo "🔍 Testing pip..."
pip --version

# Try simple package first
echo ""
echo "📦 Testing pip install with simple package..."
pip install requests --root-user-action=ignore
python -c "import requests; print('✅ requests installed successfully')"

# Check available PyTorch wheels
echo ""
echo "🔍 Checking available PyTorch versions..."
pip index versions torch --pre || echo "Could not check versions"

# Try installing PyTorch step by step
echo ""
echo "🔥 Installing PyTorch (CPU version first)..."
pip install torch --root-user-action=ignore

echo ""
echo "🔍 Testing torch import..."
python -c "
try:
    import torch
    print('✅ PyTorch imported successfully!')
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
except Exception as e:
    print(f'❌ Error importing torch: {e}')
    import sys
    print(f'Python sys.path: {sys.path}')
"

echo ""
echo "📦 Installed packages containing 'torch':"
pip list | grep -i torch || echo "No torch packages found"

echo ""
echo "🎯 If this works, try installing CUDA version:"
echo "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 --root-user-action=ignore"
