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

# Check and install required tools
echo "🔧 Checking required tools..."
if ! command -v wget &> /dev/null; then
    echo "Installing wget..."
    apt-get update && apt-get install -y wget
fi

if ! command -v unzip &> /dev/null; then
    echo "Installing unzip..."
    apt-get update && apt-get install -y unzip
fi

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
    if command -v unzip &> /dev/null; then
        unzip -q train2017.zip
    else
        echo "Using Python to extract (unzip not available)..."
        python3 -c "
import zipfile
import os
with zipfile.ZipFile('train2017.zip', 'r') as zip_ref:
    zip_ref.extractall('.')
print('✅ Extraction completed')
"
    fi
    echo "✅ Training images extracted"
else
    echo "✅ train2017 directory already exists"
fi

# Extract validation images
if [ ! -d "val2017" ]; then
    echo "📦 Extracting validation images..."
    if command -v unzip &> /dev/null; then
        unzip -q val2017.zip
    else
        echo "Using Python to extract (unzip not available)..."
        python3 -c "
import zipfile
import os
with zipfile.ZipFile('val2017.zip', 'r') as zip_ref:
    zip_ref.extractall('.')
print('✅ Extraction completed')
"
    fi
    echo "✅ Validation images extracted"
else
    echo "✅ val2017 directory already exists"
fi

# Extract annotations
if [ ! -d "annotations" ]; then
    echo "📦 Extracting annotations..."
    if command -v unzip &> /dev/null; then
        unzip -q annotations_trainval2017.zip
    else
        echo "Using Python to extract (unzip not available)..."
        python3 -c "
import zipfile
import os
with zipfile.ZipFile('annotations_trainval2017.zip', 'r') as zip_ref:
    zip_ref.extractall('.')
print('✅ Extraction completed')
"
    fi
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
