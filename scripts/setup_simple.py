"""
Simple setup script without external dependencies.
"""

import os
import json
from pathlib import Path


def create_directories():
    """Create necessary directories."""
    directories = [
        "outputs",
        "logs", 
        "cache",
        "data/demo",
        "data/demo/images",
        "checkpoints",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def create_demo_data():
    """Create simple demo data without PIL dependency."""
    print("Creating demo data structure...")
    
    # Create captions file
    captions = [
        "a red circle",
        "a blue square", 
        "a green triangle",
        "a yellow rectangle",
        "a purple oval"
    ]
    
    captions_path = Path("data/demo/captions.txt")
    with open(captions_path, "w") as f:
        for caption in captions:
            f.write(caption + "\n")
    
    # Create metadata
    metadata = {
        "dataset_name": "demo",
        "num_samples": len(captions),
        "image_size": 256,
        "created": "setup_simple.py"
    }
    
    metadata_path = Path("data/demo/metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created demo metadata with {len(captions)} samples")


def create_config_files():
    """Create configuration files."""
    
    # Simple training config
    config = {
        "model": {
            "unet": {
                "model_channels": 96,
                "channel_mult": [1, 2, 2, 2]
            }
        },
        "training": {
            "batch_size": 4,
            "max_train_steps": 1000,
            "learning_rate": 1e-4
        },
        "data": {
            "dataset_name": "data/demo",
            "image_size": 256
        }
    }
    
    # Save as JSON since we don't have yaml
    config_path = Path("configs/simple.json")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print("Created configs/simple.json")


def main():
    """Main setup function."""
    print("🎨 ScribbleDiffusion Simple Setup")
    print("==================================")
    
    create_directories()
    create_demo_data()
    create_config_files()
    
    print("\n✅ Basic setup complete!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Create demo images manually or with PIL")
    print("3. Train model: python train.py")
    print("4. Run demo: python app.py")


if __name__ == "__main__":
    main()
