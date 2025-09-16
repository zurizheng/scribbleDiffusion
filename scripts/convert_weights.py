#!/usr/bin/env python3
"""
Convert weights between linear projection formats
This handles the mismatch between use_linear_projection=True/False
"""

import torch
from safetensors.torch import load_file, save_file
import os

def convert_linear_to_conv_weights(input_path, output_path):
    """
    Convert 2D linear weights to 4D conv weights
    [dim_in, dim_out] -> [dim_out, dim_in, 1, 1]
    """
    
    print(f"🔄 Converting weights from {input_path} to {output_path}")
    
    # Load weights
    weights = load_file(input_path)
    converted_weights = {}
    
    for key, value in weights.items():
        if 'proj_in.weight' in key or 'proj_out.weight' in key:
            if len(value.shape) == 2:
                # Convert 2D to 4D: [in, out] -> [out, in, 1, 1]
                converted_value = value.transpose(0, 1).unsqueeze(-1).unsqueeze(-1).contiguous()
                converted_weights[key] = converted_value
                print(f"  ✅ {key}: {value.shape} -> {converted_value.shape}")
            else:
                converted_weights[key] = value
        else:
            converted_weights[key] = value
    
    # Save converted weights
    save_file(converted_weights, output_path)
    print(f"✅ Converted weights saved to {output_path}")

def convert_conv_to_linear_weights(input_path, output_path):
    """
    Convert 4D conv weights to 2D linear weights
    [dim_out, dim_in, 1, 1] -> [dim_in, dim_out]
    """
    
    print(f"🔄 Converting weights from {input_path} to {output_path}")
    
    # Load weights
    weights = load_file(input_path)
    converted_weights = {}
    
    for key, value in weights.items():
        if 'proj_in.weight' in key or 'proj_out.weight' in key:
            if len(value.shape) == 4 and value.shape[2] == 1 and value.shape[3] == 1:
                # Convert 4D to 2D: [out, in, 1, 1] -> [in, out]
                converted_value = value.squeeze(-1).squeeze(-1).transpose(0, 1)
                converted_weights[key] = converted_value
                print(f"  ✅ {key}: {value.shape} -> {converted_value.shape}")
            else:
                converted_weights[key] = value
        else:
            converted_weights[key] = value
    
    # Save converted weights
    save_file(converted_weights, output_path)
    print(f"✅ Converted weights saved to {output_path}")

def check_weight_shapes(weights_path):
    """Check what shapes the weights have"""
    
    weights = load_file(weights_path)
    
    print(f"📊 Weight shapes in {weights_path}:")
    
    for key, value in weights.items():
        if 'proj_in.weight' in key or 'proj_out.weight' in key:
            print(f"  {key}: {value.shape}")
    
    return weights

def main():
    """Convert our trained weights to the correct format"""
    
    model_dir = "scribble_diffusion_model"
    unet_weights_path = os.path.join(model_dir, "unet.safetensors")
    
    if not os.path.exists(unet_weights_path):
        print(f"❌ UNet weights not found at {unet_weights_path}")
        return
    
    print("🔍 Checking current weight shapes...")
    check_weight_shapes(unet_weights_path)
    
    # Convert to conv format (what the base SD model expects)
    converted_path = os.path.join(model_dir, "unet_conv_format.safetensors")
    
    print(f"\n🔄 Converting to conv format...")
    convert_linear_to_conv_weights(unet_weights_path, converted_path)
    
    print(f"\n🔍 Checking converted weight shapes...")
    check_weight_shapes(converted_path)

if __name__ == "__main__":
    main()