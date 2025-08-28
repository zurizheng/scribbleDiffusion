#!/bin/bash
# Simple diffusers test installation

echo "🔧 Testing Diffusers Installation Step by Step"
echo "=============================================="

# Ensure we're in the venv
source .venv/bin/activate

echo "📍 Environment check:"
echo "   Python: $(which python)"
echo "   Pip: $(which pip)"
echo "   Virtual env: $VIRTUAL_ENV"

echo ""
echo "🧪 Step 1: Try installing diffusers directly..."
python -m pip install diffusers==0.18.2 --no-deps --verbose

echo ""
echo "🧪 Step 2: Check if it was installed..."
python -m pip list | grep diffusers

echo ""
echo "🧪 Step 3: Try importing diffusers..."
python -c "
try:
    import diffusers
    print('✅ Diffusers imported successfully!')
    print(f'Version: {diffusers.__version__}')
    print(f'Location: {diffusers.__file__}')
except Exception as e:
    print(f'❌ Import failed: {e}')
    import sys
    print(f'Python path: {sys.path}')
"

echo ""
echo "🧪 Step 4: Install dependencies separately..."
python -m pip install huggingface-hub
python -m pip install regex
python -m pip install requests
python -m pip install Pillow
python -m pip install safetensors

echo ""
echo "🧪 Step 5: Try importing again..."
python -c "
try:
    import diffusers
    print('✅ Diffusers imported successfully after dependencies!')
    from diffusers import AutoencoderKL
    print('✅ AutoencoderKL imported successfully!')
except Exception as e:
    print(f'❌ Still failing: {e}')
"
