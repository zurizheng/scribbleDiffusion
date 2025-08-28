#!/bin/bash
# Training launcher script that ensures proper Python path

echo "🚀 Starting ScribbleDiffusion Training"
echo "======================================"

# Ensure we're in the project root
cd /workspace/scribbleDiffusion

# Set Python path FIRST before any Python operations
export PYTHONPATH="/workspace/scribbleDiffusion:$PYTHONPATH"

# Activate virtual environment
source .venv/bin/activate

# Verify environment
echo "📍 Environment check:"
echo "   Working dir: $(pwd)"
echo "   Python: $(which python)"
echo "   Virtual env: $VIRTUAL_ENV"
echo "   Python path: $PYTHONPATH"

echo ""
echo "🔍 Quick import test..."
python -c "
import sys
print('Python sys.path:')
for p in sys.path[:3]:
    print(f'  {p}')
print('  ...')

try:
    from src.models.unet import SketchConditionedUNet
    from src.models.hint_encoder import HintEncoder
    print('✅ All modules imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

echo ""
echo "🎯 Starting training with RTX 3090 config..."
echo "Config: configs/rtx3090.yaml"
echo ""

# Run training with proper Python path
python train.py --config configs/rtx3090.yaml "$@"
