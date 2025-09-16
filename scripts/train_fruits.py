#!/usr/bin/env python3
"""
Quick training script for fruit dataset
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.utils.device_utils import get_optimal_device, get_device_info

def main():
    print("🍎 Starting Fruit ScribbleDiffusion Training")
    print("=" * 50)
    
    # Check device
    device = get_optimal_device()
    device_info = get_device_info()
    
    print(f"Device: {device}")
    print(f"Memory: {device_info.get('total_memory_gb', 'Unknown')} GB")
    print()
    
    # Check if fruit dataset exists
    fruit_dir = Path("data/fruits")
    if not fruit_dir.exists():
        print(f"❌ Fruit dataset not found at {fruit_dir}")
        print("Please ensure your fruit dataset is in data/fruits/")
        return
    
    # Count fruit folders
    fruit_folders = [d for d in fruit_dir.iterdir() if d.is_dir()]
    print(f"Found {len(fruit_folders)} fruit folders:")
    for folder in sorted(fruit_folders):
        image_count = len(list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg")))
        print(f"  {folder.name}: {image_count} images")
    print()
    
    # Test dataset loading
    print("Testing fruit dataset loading...")
    try:
        from src.data.fruit_dataset import FruitDataset
        dataset = FruitDataset(str(fruit_dir), image_size=256)
        print(f"✅ Successfully loaded {len(dataset)} fruit images")
        
        # Test a sample
        sample = dataset[0]
        print(f"Sample fruit: {sample['fruit_label']}")
        print(f"Text prompt: {sample['text_prompt']}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Sketch shape: {sample['sketch'].shape}")
        print()
        
    except Exception as e:
        print(f"❌ Error loading fruit dataset: {e}")
        return
    
    # Start training
    print("Starting training...")
    print("Command: python scripts/train.py --config configs/fruit_training.yaml")
    print()
    
    import subprocess
    try:
        subprocess.run([
            sys.executable, "scripts/train.py", 
            "--config", "configs/fruit_training.yaml"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")

if __name__ == "__main__":
    main()