#!/bin/bash
# Complete pipeline: Train → Export → Upload to HuggingFace

set -e  # Exit on any error

echo "🚀 ScribbleDiffusion: Train → Export → Upload Pipeline"
echo "=================================================="

# Configuration
MODEL_NAME="scribble-diffusion-cross-attention-v1"
HF_USERNAME="yourusername"  # Change this!
CHECKPOINT_STEPS="1000"

# Phase 1: Training
echo "📚 Phase 1: Training with cached edges..."
python train_cached.py --config configs/rtx3090_cached.yaml

# Phase 2: Export to SafeTensors
echo "📦 Phase 2: Exporting model..."
python scripts/export_model.py \
  --checkpoint "models/checkpoint-${CHECKPOINT_STEPS}/pytorch_model.bin" \
  --output_dir "exported_models/${MODEL_NAME}" \
  --format safetensors

# Phase 3: Upload to HuggingFace
echo "🤗 Phase 3: Uploading to Hugging Face..."
python scripts/upload_to_hf.py \
  --model_path "exported_models/${MODEL_NAME}" \
  --repo_name "${HF_USERNAME}/${MODEL_NAME}" \
  --private  # Remove this flag to make public

echo "✅ Pipeline complete!"
echo "🌐 Model available at: https://huggingface.co/${HF_USERNAME}/${MODEL_NAME}"
echo ""
echo "📝 Next steps:"
echo "1. Update README.md with model card"
echo "2. Test model download and inference"
echo "3. Share with community!"