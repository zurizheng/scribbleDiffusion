"""
Setup script to download pretrained models and prepare demo data.
"""

import os
import requests
from pathlib import Path
import zipfile
import tarfile
from PIL import Image, ImageDraw
import random
import json


def download_file(url: str, destination: str, chunk_size: int = 8192):
    """Download a file from URL to destination."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
    
    print(f"Downloaded {destination}")


def setup_directories():
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


def create_demo_dataset():
    """Create a small demo dataset for testing."""
    print("Creating demo dataset...")
    
    output_dir = Path("data/demo")
    images_dir = output_dir / "images"
    
    # Create simple geometric shapes with captions
    shapes_data = []
    colors = ["red", "blue", "green", "yellow", "purple", "orange"]
    shapes = ["circle", "square", "triangle", "rectangle"]
    
    for i in range(50):  # Create 50 demo samples
        # Create image
        img = Image.new("RGB", (256, 256), "white")
        draw = ImageDraw.Draw(img)
        
        # Pick random shape and color
        color = random.choice(colors)
        shape = random.choice(shapes)
        
        # Draw shape
        if shape == "circle":
            x, y = random.randint(50, 200), random.randint(50, 200)
            r = random.randint(20, 60)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=color, outline="black", width=2)
            
        elif shape == "square":
            x, y = random.randint(50, 150), random.randint(50, 150)
            size = random.randint(40, 100)
            draw.rectangle([x, y, x+size, y+size], fill=color, outline="black", width=2)
            
        elif shape == "triangle":
            x, y = random.randint(80, 180), random.randint(80, 180)
            size = random.randint(30, 80)
            points = [(x, y-size), (x-size, y+size), (x+size, y+size)]
            draw.polygon(points, fill=color, outline="black", width=2)
            
        elif shape == "rectangle":
            x, y = random.randint(50, 120), random.randint(50, 120)
            w, h = random.randint(60, 120), random.randint(40, 80)
            draw.rectangle([x, y, x+w, y+h], fill=color, outline="black", width=2)
        
        # Save image
        img_path = images_dir / f"demo_{i:03d}.png"
        img.save(img_path)
        
        # Create caption
        caption = f"a {color} {shape}"
        shapes_data.append({
            "image": str(img_path.name),
            "caption": caption,
        })
    
    # Save captions file
    captions_file = output_dir / "captions.txt"
    with open(captions_file, "w") as f:
        for item in shapes_data:
            f.write(item["caption"] + "\n")
    
    # Save metadata
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(shapes_data, f, indent=2)
    
    print(f"Created demo dataset with {len(shapes_data)} samples")


def download_pretrained_models():
    """Download pretrained model components."""
    print("Setting up pretrained model components...")
    
    # Note: In practice, you'd download actual pretrained weights
    # For now, we'll just create placeholder files
    
    cache_dir = Path("cache")
    
    # Create placeholder model files
    placeholder_files = [
        "vae_config.json",
        "text_encoder_config.json", 
        "unet_config.json",
        "scheduler_config.json",
    ]
    
    for filename in placeholder_files:
        placeholder_path = cache_dir / filename
        with open(placeholder_path, "w") as f:
            json.dump({"placeholder": True}, f)
        print(f"Created placeholder: {filename}")
    
    print("Note: Replace placeholder files with actual pretrained models")


def setup_environment():
    """Set up the complete environment."""
    print("Setting up ScribbleDiffusion environment...")
    
    # Create directories
    setup_directories()
    
    # Create demo dataset
    create_demo_dataset()
    
    # Download pretrained models (placeholder)
    download_pretrained_models()
    
    # Create example config files
    create_example_configs()
    
    print("\n✅ Environment setup complete!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Train model: python train.py --config configs/base.yaml")
    print("3. Run demo: python app.py")


def create_example_configs():
    """Create additional example configuration files."""
    
    # Fast training config for testing
    fast_config = {
        "model": {
            "unet": {
                "model_channels": 96,  # Smaller model
                "channel_mult": [1, 2, 2, 2],
            }
        },
        "training": {
            "batch_size": 4,
            "max_train_steps": 1000,  # Fewer steps
            "validation_steps": 200,
            "save_interval": 500,
        },
        "data": {
            "dataset_name": "data/demo",  # Use demo dataset
            "image_size": 256,
        }
    }
    
    import yaml
    with open("configs/fast.yaml", "w") as f:
        yaml.dump(fast_config, f, default_flow_style=False)
    
    print("Created configs/fast.yaml for quick testing")


def main():
    """Main setup function."""
    setup_environment()


if __name__ == "__main__":
    main()
