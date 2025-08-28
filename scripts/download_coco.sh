#!/bin/bash
# COCO Dataset Download Script for ScribbleDiffusion
# Downloads COCO 2017 training images and annotations

set -e  # Exit on any error

echo "🖼️ Downloading COCO 2017 Dataset for ScribbleDiffusion"
echo "=" * 60

# Get the project root directory (one level up from scripts)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COCO_DIR="$PROJECT_ROOT/data/coco"

# Create and navigate to COCO directory
mkdir -p "$COCO_DIR"
cd "$COCO_DIR"

# Check available disk space (COCO is ~25GB)
echo "📊 Checking disk space..."
df -h .

# Download COCO 2017 training images (~18GB)
if [ ! -f "train2017.zip" ]; then
    echo "📥 Downloading COCO 2017 training images (~18GB)..."
    wget --progress=bar:force http://images.cocodataset.org/zips/train2017.zip
else
    echo "✅ train2017.zip already exists"
fi

# Download COCO 2017 validation images (~1GB) - useful for validation
if [ ! -f "val2017.zip" ]; then
    echo "📥 Downloading COCO 2017 validation images (~1GB)..."
    wget --progress=bar:force http://images.cocodataset.org/zips/val2017.zip
else
    echo "✅ val2017.zip already exists"
fi

# Download COCO 2017 annotations (~241MB)
if [ ! -f "annotations_trainval2017.zip" ]; then
    echo "📥 Downloading COCO 2017 annotations (~241MB)..."
    wget --progress=bar:force http://images.cocodataset.org/annotations/annotations_trainval2017.zip
else
    echo "✅ annotations_trainval2017.zip already exists"
fi

# Extract training images
if [ ! -d "train2017" ]; then
    echo "📦 Extracting training images..."
    unzip -q train2017.zip
    echo "✅ Training images extracted"
else
    echo "✅ train2017 directory already exists"
fi

# Extract validation images
if [ ! -d "val2017" ]; then
    echo "📦 Extracting validation images..."
    unzip -q val2017.zip
    echo "✅ Validation images extracted"
else
    echo "✅ val2017 directory already exists"
fi

# Extract annotations
if [ ! -d "annotations" ]; then
    echo "📦 Extracting annotations..."
    unzip -q annotations_trainval2017.zip
    echo "✅ Annotations extracted"
else
    echo "✅ annotations directory already exists"
fi

# Clean up zip files to save space (optional)
echo ""
read -p "🗑️ Delete zip files to save ~19GB of space? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f train2017.zip val2017.zip annotations_trainval2017.zip
    echo "✅ Zip files deleted"
else
    echo "📦 Zip files kept for future use"
fi

# Verify download
echo ""
echo "📊 Verifying download..."
if [ -d "train2017" ] && [ -d "annotations" ]; then
    train_count=$(ls train2017/ | wc -l)
    val_count=$(ls val2017/ 2>/dev/null | wc -l)
    echo "✅ Found $train_count training images"
    echo "✅ Found $val_count validation images"
    
    if [ -f "annotations/instances_train2017.json" ]; then
        echo "✅ Instance annotations found"
    fi
    
    if [ -f "annotations/captions_train2017.json" ]; then
        echo "✅ Caption annotations found"
    fi
    
    echo ""
    echo "🎉 COCO dataset download completed successfully!"
    echo ""
    echo "Dataset structure:"
    echo "  data/coco/train2017/          - Training images ($train_count files)"
    echo "  data/coco/val2017/            - Validation images ($val_count files)"
    echo "  data/coco/annotations/        - Annotation files"
    echo ""
    echo "🚀 Ready to start training with real COCO data!"
    echo "   Use: python train.py --config configs/rtx3090.yaml"
    echo ""
    
else
    echo "❌ Download verification failed"
    exit 1
fi
