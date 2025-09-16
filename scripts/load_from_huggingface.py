#!/usr/bin/env python3
"""
Load and use ScribbleDiffusion model from Hugging Face
Example: zurizheng/scribble-diffusion-v1
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel

# Import device utilities
from src.utils.device_utils import get_optimal_device
from transformers import CLIPTokenizer, CLIPTextModel

# Import your custom models (you'll need these in your local environment)
try:
    from src.models.unet import UNet2DConditionModel
    from src.models.sketch_encoder import SketchCrossAttentionEncoder
except ImportError:
    print("⚠️ Custom models not found. You'll need the src/ directory from the training repo.")
    print("For now, we'll show you how to download the weights.")

class ScribbleDiffusionPipeline:
    def __init__(self, model_id="zurizheng/scribble-diffusion-v1"):
        """Load ScribbleDiffusion model from Hugging Face"""
        
        self.device = get_optimal_device()
        print(f"🚀 Loading ScribbleDiffusion from {model_id}")
        print(f"📱 Using device: {self.device}")
        
        # Download model weights
        print("📥 Downloading UNet...")
        unet_path = hf_hub_download(model_id, "unet.safetensors")
        
        print("📥 Downloading SketchEncoder...")
        sketch_encoder_path = hf_hub_download(model_id, "sketch_encoder.safetensors")
        
        # Load base Stable Diffusion components
        print("📥 Loading base SD components...")
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
        self.tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
        self.scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        
        # Initialize custom models (you'll need the model definitions)
        try:
            print("🔧 Initializing custom UNet...")
            self.unet = UNet2DConditionModel(
                sample_size=512,
                in_channels=4,
                out_channels=4,
                layers_per_block=2,
                block_out_channels=[320, 640, 1280, 1280],
                down_block_types=[
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D", 
                    "CrossAttnDownBlock2D",
                    "DownBlock2D"
                ],
                up_block_types=[
                    "UpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D"
                ],
                cross_attention_dim=768,
                attention_head_dim=8,
                use_linear_projection=True
            )
            
            print("🔧 Initializing SketchEncoder...")
            self.sketch_encoder = SketchCrossAttentionEncoder(
                cross_attention_dim=768,
                attention_head_dim=8,
                num_attention_heads=12
            )
            
            # Load trained weights
            print("⚡ Loading trained weights...")
            unet_weights = load_file(unet_path)
            sketch_encoder_weights = load_file(sketch_encoder_path)
            
            self.unet.load_state_dict(unet_weights)
            self.sketch_encoder.load_state_dict(sketch_encoder_weights)
            
            # Move to device
            self.unet.to(self.device)
            self.sketch_encoder.to(self.device)
            self.vae.to(self.device)
            self.text_encoder.to(self.device)
            
            print("✅ ScribbleDiffusion loaded successfully!")
            
        except NameError:
            print("❌ Custom model classes not available.")
            print("📋 You have the weights downloaded to:")
            print(f"   UNet: {unet_path}")
            print(f"   SketchEncoder: {sketch_encoder_path}")
            print("💡 Copy the src/ directory from training repo to use these weights.")
            
    def preprocess_sketch(self, sketch_image):
        """Preprocess sketch for the model"""
        # Convert to grayscale if needed
        if len(sketch_image.shape) == 3:
            sketch_image = cv2.cvtColor(sketch_image, cv2.COLOR_RGB2GRAY)
        
        # Resize to 512x512
        sketch_image = cv2.resize(sketch_image, (512, 512))
        
        # Apply edge detection if needed
        if sketch_image.max() > 1:  # If not already edges
            sketch_image = cv2.Canny(sketch_image, 50, 150)
        
        # Convert to tensor
        sketch_tensor = torch.from_numpy(sketch_image).float().unsqueeze(0).unsqueeze(0) / 255.0
        return sketch_tensor.to(self.device)
    
    def generate(self, sketch, prompt, num_steps=20, guidance_scale=7.5):
        """Generate image from sketch and text prompt"""
        
        if not hasattr(self, 'unet'):
            print("❌ Model not properly loaded. Need custom model definitions.")
            return None
        
        print(f"🎨 Generating: '{prompt}'")
        
        # Preprocess sketch
        sketch_tensor = self.preprocess_sketch(sketch)
        
        # Encode text
        text_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
        
        # Process sketch
        sketch_features = self.sketch_encoder(sketch_tensor)
        
        # Generate latents
        latents = torch.randn(1, 4, 64, 64).to(self.device)
        
        # Denoising loop (simplified)
        self.scheduler.set_timesteps(num_steps)
        
        for t in self.scheduler.timesteps:
            # Predict noise
            noise_pred = self.unet(latents, t, encoder_hidden_states=text_embeddings)
            
            # Update latents
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode to image
        with torch.no_grad():
            images = self.vae.decode(latents / 0.18215).sample
            images = (images / 2 + 0.5).clamp(0, 1)
            images = images.cpu().permute(0, 2, 3, 1).numpy()
            images = (images * 255).astype(np.uint8)
        
        return Image.fromarray(images[0])

def quick_download_example():
    """Quick example of downloading weights"""
    
    model_id = "zurizheng/scribble-diffusion-v1"
    
    print(f"📦 Downloading from {model_id}")
    
    # Download files
    files_to_download = ["unet.safetensors", "sketch_encoder.safetensors", "config.json", "README.md"]
    
    downloaded_files = {}
    for filename in files_to_download:
        try:
            file_path = hf_hub_download(model_id, filename)
            downloaded_files[filename] = file_path
            print(f"✅ {filename}: {file_path}")
        except Exception as e:
            print(f"❌ {filename}: {e}")
    
    return downloaded_files

if __name__ == "__main__":
    # Quick download example
    print("🚀 ScribbleDiffusion Hugging Face Loader")
    print("=" * 50)
    
    # Option 1: Just download the files
    print("📥 Option 1: Download model files")
    files = quick_download_example()
    
    print("\n🔧 Option 2: Full pipeline (requires src/ directory)")
    print("Copy the src/ directory from training repo, then:")
    print("pipeline = ScribbleDiffusionPipeline('zurizheng/scribble-diffusion-v1')")
    
    print("\n💡 Next steps:")
    print("1. Copy src/ directory from training repo")
    print("2. Install requirements: pip install diffusers transformers safetensors")
    print("3. Load and use the pipeline!")