#!/usr/bin/env python3
"""
Export trained ScribbleDiffusion model for deployment
Converts PyTorch checkpoint to SafeTensors format for Hugging Face
"""

import argparse
import torch
import os
from pathlib import Path
import json
from safetensors.torch import save_file
import sys

def export_model(checkpoint_path, output_dir, format_type="safetensors"):
    """Export trained model to deployment format"""
    
    print(f"🔄 Exporting model from {checkpoint_path}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print("📥 Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model state dicts
    unet_state_dict = checkpoint.get('unet_state_dict', {})
    hint_encoder_state_dict = checkpoint.get('hint_encoder_state_dict', {})
    
    if not unet_state_dict:
        print("❌ No UNet state dict found in checkpoint")
        return False
        
    if not hint_encoder_state_dict:
        print("❌ No HintEncoder state dict found in checkpoint")
        return False
    
    # Save models in requested format
    if format_type == "safetensors":
        print("💾 Saving UNet as SafeTensors...")
        save_file(unet_state_dict, output_dir / "unet.safetensors")
        
        print("💾 Saving HintEncoder as SafeTensors...")
        save_file(hint_encoder_state_dict, output_dir / "hint_encoder.safetensors")
        
    elif format_type == "pytorch":
        print("💾 Saving UNet as PyTorch...")
        torch.save(unet_state_dict, output_dir / "unet.pth")
        
        print("💾 Saving HintEncoder as PyTorch...")
        torch.save(hint_encoder_state_dict, output_dir / "hint_encoder.pth")
    
    # Save model configuration
    config = {
        "model_type": "scribble_diffusion",
        "architecture": {
            "unet": {
                "in_channels": 4,
                "out_channels": 4,
                "model_channels": checkpoint.get('config', {}).get('model_channels', 320),
                "attention_resolutions": [1, 2, 4, 8],
                "num_res_blocks": 2,
                "channel_mult": [1, 2, 4, 4],
                "num_heads": 8,
                "use_spatial_transformer": True,
                "transformer_depth": 2,
                "context_dim": 768
            },
            "hint_encoder": {
                "in_channels": 1,
                "hint_channels": [64, 128, 256, 512],
                "injection_layers": [0, 1, 2, 3],
                "injection_method": "add"
            }
        },
        "training_info": {
            "steps_trained": checkpoint.get('step', 0),
            "epoch": checkpoint.get('epoch', 0),
            "loss": checkpoint.get('loss', 0.0),
            "hardware": "RTX 3090 24GB",
            "framework": "PyTorch + Diffusers"
        }
    }
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create model card
    model_card = f"""---
license: apache-2.0
library_name: diffusers
tags:
- stable-diffusion
- image-generation
- sketch-to-image
- controlnet
- pytorch
---

# ScribbleDiffusion - Sketch to Image Generation

This model was trained on RTX 3090 24GB VRAM using the ScribbleDiffusion architecture.

## Model Description

ScribbleDiffusion is a lightweight ControlNet variant that generates high-quality images from simple sketches. It uses a custom UNet with hint injection for sketch conditioning.

## Architecture

- **UNet**: {config['architecture']['unet']['model_channels']} base channels with attention
- **HintEncoder**: Lightweight sketch encoder with {len(config['architecture']['hint_encoder']['hint_channels'])} levels
- **Training Steps**: {config['training_info']['steps_trained']:,}

## Usage

```python
import torch
from src.models.unet import SketchConditionedUNet
from src.models.hint_encoder import HintEncoder

# Load models
unet = SketchConditionedUNet.from_pretrained("./unet.safetensors")
hint_encoder = HintEncoder.from_pretrained("./hint_encoder.safetensors")

# Generate image from sketch
# (Add your inference code here)
```

## Training Details

- **Hardware**: RTX 3090 24GB VRAM
- **Framework**: PyTorch + Diffusers
- **Dataset**: COCO 2017
- **Resolution**: 512x512
- **Batch Size**: 12

## License

Apache 2.0
"""
    
    with open(output_dir / "README.md", 'w') as f:
        f.write(model_card)
    
    # Calculate model sizes
    unet_size = os.path.getsize(output_dir / f"unet.{format_type.split('_')[0]}")
    hint_size = os.path.getsize(output_dir / f"hint_encoder.{format_type.split('_')[0]}")
    total_size = unet_size + hint_size
    
    print(f"✅ Export completed!")
    print(f"📊 Model sizes:")
    print(f"   UNet: {unet_size / 1e6:.1f} MB")
    print(f"   HintEncoder: {hint_size / 1e6:.1f} MB")
    print(f"   Total: {total_size / 1e6:.1f} MB")
    print(f"📁 Saved to: {output_dir}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Export ScribbleDiffusion model")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--format", choices=["safetensors", "pytorch"], 
                       default="safetensors", help="Export format")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    success = export_model(args.checkpoint, args.output, args.format)
    
    if success:
        print(f"🎉 Ready for deployment!")
        print(f"💡 Next steps:")
        print(f"   1. Upload to Hugging Face: python scripts/upload_to_hf.py --model_path {args.output}")
        print(f"   2. Or compress for sharing: tar -czf scribble_diffusion_model.tar.gz {args.output}")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
