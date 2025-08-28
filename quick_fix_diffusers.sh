#!/bin/bash
# Quick fix for huggingface_hub compatibility issue
# Run this instead of recreating the whole .venv

echo "🔧 Quick Fix: HuggingFace Hub Compatibility"
echo "=========================================="

# Activate existing venv
source .venv/bin/activate

echo "📍 Current environment:"
echo "   Python: $(which python)"
echo "   Pip: $(which pip)"

echo ""
echo "🔍 Current versions:"
python -c "
try:
    import huggingface_hub
    print(f'HuggingFace Hub: {huggingface_hub.__version__}')
except:
    print('HuggingFace Hub: not installed')

try:
    import diffusers
    print(f'Diffusers: {diffusers.__version__}')
except:
    print('Diffusers: not installed')
"

echo ""
echo "🔄 Fixing compatibility..."

# Only fix the problematic packages
echo "Downgrading huggingface_hub to compatible version..."
python -m pip install huggingface_hub==0.16.4 --force-reinstall

echo "Ensuring diffusers is correct version..."
python -m pip install diffusers==0.18.2 --force-reinstall

echo "Fixing safetensors version..."
python -m pip install safetensors==0.3.1 --force-reinstall

echo ""
echo "✅ Testing the fix..."
python -c "
try:
    print('Testing huggingface_hub...')
    from huggingface_hub import cached_download, hf_hub_download
    print('✅ huggingface_hub functions available')
    
    print('Testing diffusers...')
    import diffusers
    print(f'✅ Diffusers {diffusers.__version__} imported')
    
    from diffusers import AutoencoderKL
    print('✅ AutoencoderKL imported successfully')
    
    print('✅ All compatibility issues fixed!')
    
except Exception as e:
    print(f'❌ Still have issues: {e}')
"

echo ""
echo "🎉 Quick fix completed!"
echo "Your environment should now work without rebuilding everything."
