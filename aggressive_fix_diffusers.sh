#!/bin/bash
# Aggressive fix for huggingface_hub compatibility
# Completely removes and reinstalls problematic packages

echo "🔧 Aggressive Fix: Complete Package Removal and Reinstall"
echo "========================================================="

# Activate existing venv
source .venv/bin/activate

echo "📍 Current environment:"
echo "   Python: $(which python)"
echo "   Pip: $(which pip)"

echo ""
echo "🗑️ Completely removing problematic packages..."

# Uninstall all related packages completely
python -m pip uninstall huggingface_hub -y
python -m pip uninstall diffusers -y
python -m pip uninstall safetensors -y
python -m pip uninstall tokenizers -y  # Often causes conflicts too

# Clear pip cache to ensure fresh downloads
python -m pip cache purge

echo ""
echo "🧹 Cleaning up any leftover files..."
# Remove any leftover package directories
rm -rf .venv/lib/python3.10/site-packages/huggingface_hub*
rm -rf .venv/lib/python3.10/site-packages/diffusers*
rm -rf .venv/lib/python3.10/site-packages/safetensors*

echo ""
echo "📦 Installing compatible versions from scratch..."

# Install in specific order with exact versions
echo "Installing huggingface_hub 0.16.4..."
python -m pip install huggingface_hub==0.16.4 --no-cache-dir

echo "Installing tokenizers..."
python -m pip install tokenizers==0.13.3 --no-cache-dir

echo "Installing safetensors..."
python -m pip install safetensors==0.3.1 --no-cache-dir

echo "Installing diffusers 0.18.2..."
python -m pip install diffusers==0.18.2 --no-cache-dir

echo ""
echo "📋 Verifying installed versions..."
python -c "
import subprocess
result = subprocess.run(['python', '-m', 'pip', 'list'], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if any(pkg in line.lower() for pkg in ['huggingface', 'diffusers', 'safetensors', 'tokenizers']):
        print(f'   {line}')
"

echo ""
echo "✅ Testing the aggressive fix..."
python -c "
try:
    print('Testing huggingface_hub imports...')
    from huggingface_hub import cached_download, hf_hub_download, HfFolder
    print('✅ All huggingface_hub functions available')
    
    print('Testing diffusers...')
    import diffusers
    print(f'✅ Diffusers {diffusers.__version__} imported')
    
    from diffusers import AutoencoderKL
    print('✅ AutoencoderKL imported successfully')
    
    print('🧪 Testing actual usage...')
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[64, 128],
        latent_channels=4
    )
    print('✅ AutoencoderKL created successfully!')
    
    print('')
    print('🎉 ALL COMPATIBILITY ISSUES FIXED!')
    print('🚀 Ready to start training!')
    
except Exception as e:
    print(f'❌ Still have issues: {e}')
    print('')
    print('🔍 Debug info:')
    import traceback
    traceback.print_exc()
"

echo ""
echo "🎯 Aggressive fix completed!"
