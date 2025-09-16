#!/bin/bash
# RunPod setup script for ScribbleDiffusion fruit training

echo "🍎 Setting up ScribbleDiffusion for RunPod training"
echo "=================================================="

# Update system
apt-get update
apt-get install -y git wget unzip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate safetensors
pip install opencv-python pillow numpy omegaconf tensorboard
pip install tqdm datasets

# Verify GPU
echo "GPU Information:"
nvidia-smi

# Test PyTorch CUDA
python3 -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

# Test fruit setup if dataset exists
if [ -d "data/fruits" ]; then
    echo "Testing fruit dataset..."
    python3 test_fruit_setup.py
else
    echo "⚠️  Upload your fruit dataset to data/fruits/ before training"
    echo "Expected structure:"
    echo "  data/fruits/Apple_Good/"
    echo "  data/fruits/Banana_Good/"
    echo "  data/fruits/Guava_Good/"
    echo "  data/fruits/Lime_Good/"
    echo "  data/fruits/Orange_Good/"
    echo "  data/fruits/Pomegranate_Good/"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Upload your fruit dataset to data/fruits/"
echo "2. Run: python3 scripts/train.py --config configs/fruit_training.yaml"
echo "3. Monitor with: tensorboard --logdir outputs/fruit_model/logs --host 0.0.0.0 --port 6006"