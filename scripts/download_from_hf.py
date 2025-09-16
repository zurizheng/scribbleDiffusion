#!/usr/bin/env python3
"""
Simple script to download ScribbleDiffusion model from Hugging Face
"""

from huggingface_hub import snapshot_download
import os

def download_scribble_diffusion(model_id="zurizheng/scribble-diffusion-v1", local_dir="./scribble_diffusion_model"):
    """Download the entire model repository"""
    
    print(f"📦 Downloading {model_id} to {local_dir}")
    
    # Download all files
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    
    print(f"✅ Model downloaded to: {local_dir}")
    print(f"📁 Contents:")
    
    for file in os.listdir(local_dir):
        file_path = os.path.join(local_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"   {file}: {size:.1f} MB")

if __name__ == "__main__":
    download_scribble_diffusion()