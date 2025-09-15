#!/bin/bash
# Transfer script for moving model from RunPod to local

echo "🚀 Transferring ScribbleDiffusion model to local development"

# Create local directories
mkdir -p ~/scribble_local/models
mkdir -p ~/scribble_local/cache

# Transfer essential files
echo "📦 Downloading model weights..."
scp -r runpod:/workspace/scribbleDiffusion/models/checkpoint-* ~/scribble_local/models/

echo "💾 Downloading small edge cache..."
scp -r runpod:/workspace/scribbleDiffusion/data/coco/edges_cache/cache_metadata.json ~/scribble_local/cache/

echo "🔧 Downloading configs..."
scp runpod:/workspace/scribbleDiffusion/configs/local_8gb.yaml ~/scribble_local/

echo "📋 Memory usage estimates for 8GB VRAM:"
echo "- Model: ~2-3GB"
echo "- Training batch: ~1-2GB" 
echo "- OS/other: ~1GB"
echo "- Available: ~3-4GB buffer"
echo ""
echo "✅ Ready for local development!"
echo "Next: cd ~/scribble_local && python train_cached.py --config local_8gb.yaml"