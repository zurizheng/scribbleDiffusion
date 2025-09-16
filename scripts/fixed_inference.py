#!/usr/bin/env python3
"""
Fixed ScribbleDiffusion inference that works without the missing SketchTextCombiner
Since the combiner just did concatenation, we can implement this directly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from safetensors.torch import load_file
from tqdm import tqdm

# Use diffusers for base components
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel

# Import our custom models
from models.sketch_encoder import SketchCrossAttentionEncoder
from utils.device_utils import get_optimal_device

class FixedScribblePipeline:
    def __init__(self, model_path="scribble_diffusion_model", force_cpu=True):
        """Fixed pipeline that works without SketchTextCombiner"""
        
        print("🚀 Loading ScribbleDiffusion (Fixed - no combiner needed)...")
        
        # Use optimal device detection
        if force_cpu:
            self.device = torch.device("cpu")
            self.dtype = torch.float32
        else:
            self.device = get_optimal_device()
            self.dtype = torch.float32
            
        print(f"📱 Using device: {self.device}")
        print(f"🔢 Using dtype: {self.dtype}")
        
        # Load base components
        print("📦 Loading base components...")
        
        # Text encoder and tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="text_encoder",
            torch_dtype=self.dtype
        ).to(self.device)
        
        # VAE for latent encoding/decoding
        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="vae",
            torch_dtype=self.dtype
        ).to(self.device)
        
        # Scheduler
        self.scheduler = DDIMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
        )
        
        # Load custom UNet
        print("⚡ Loading custom UNet...")
        self.unet = self._load_custom_unet(model_path)
        
        # Load sketch encoder  
        print("🎨 Loading sketch encoder...")
        self.sketch_encoder = self._load_sketch_encoder(model_path)
        
        # Set to eval mode
        self.text_encoder.eval()
        self.vae.eval()
        self.unet.eval()
        self.sketch_encoder.eval()
        
        print("✅ Pipeline ready!")
    
    def _load_custom_unet(self, model_path):
        """Load UNet with correct architecture"""
        
        # Create UNet with correct architecture (use_linear_projection=False for conv weights)
        unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            subfolder="unet",
            torch_dtype=self.dtype,
            use_linear_projection=False,  # Match trained architecture
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True
        ).to(self.device)
        
        # Load the trained weights (converted format)
        try:
            unet_weights_path = f"{model_path}/unet_conv_format.safetensors"
            if os.path.exists(unet_weights_path):
                unet_weights = load_file(unet_weights_path)
                print("📦 Using converted conv format weights")
            else:
                unet_weights_path = f"{model_path}/unet.safetensors"
                unet_weights = load_file(unet_weights_path)
                print("📦 Using original weights")
            
            # Convert weights to correct dtype
            unet_weights = {k: v.to(self.dtype) for k, v in unet_weights.items()}
            
            # Load weights
            missing_keys, unexpected_keys = unet.load_state_dict(unet_weights, strict=False)
            
            if not missing_keys and not unexpected_keys:
                print("✅ UNet weights loaded perfectly")
            else:
                print(f"⚠️ UNet loaded with {len(missing_keys)} missing, {len(unexpected_keys)} extra keys")
            
        except Exception as e:
            print(f"❌ Failed to load UNet weights: {e}")
            print("   Using base SD weights instead")
        
        return unet
    
    def _load_sketch_encoder(self, model_path):
        """Load sketch encoder"""
        
        # Create sketch encoder with same config as training
        sketch_encoder = SketchCrossAttentionEncoder(
            in_channels=1,
            hidden_dim=512,
            num_sketch_tokens=77,  # Match text token length
            cross_attention_dim=768
        ).to(self.device).to(self.dtype)
        
        # Load weights
        try:
            sketch_weights = load_file(f"{model_path}/sketch_encoder.safetensors")
            sketch_weights = {k: v.to(self.dtype) for k, v in sketch_weights.items()}
            sketch_encoder.load_state_dict(sketch_weights)
            print("✅ Sketch encoder weights loaded")
        except Exception as e:
            print(f"❌ Failed to load sketch encoder: {e}")
            print("   Will use random initialization")
        
        return sketch_encoder
    
    def encode_prompt(self, prompt):
        """Encode text prompt to embeddings"""
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
        
        return text_embeddings
    
    def preprocess_sketch(self, sketch_path):
        """Process sketch image to edge map"""
        
        # Load and convert to grayscale
        if isinstance(sketch_path, str):
            sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
        else:
            sketch = np.array(sketch_path.convert('L'))
        
        # Resize to 512x512
        sketch = cv2.resize(sketch, (512, 512))
        
        # Apply edge detection if needed
        if sketch.max() > 1:
            edges = cv2.Canny(sketch, 50, 150)
        else:
            edges = sketch
        
        # Convert to tensor
        edges = torch.from_numpy(edges.astype(np.float32) / 255.0)
        edges = edges.unsqueeze(0).unsqueeze(0)  # [1, 1, 512, 512]
        
        return edges.to(self.device).to(self.dtype)
    
    def combine_embeddings(self, text_embeddings, sketch_features):
        """
        Implement the missing SketchTextCombiner functionality
        The combiner just did concatenation: torch.cat([text_embeddings, sketch_embeddings], dim=1)
        """
        # text_embeddings: [B, 77, 768]
        # sketch_features: [B, 77, 768] 
        
        # Simple concatenation along sequence dimension (what the combiner did)
        combined = torch.cat([text_embeddings, sketch_features], dim=1)
        # Result: [B, 154, 768]
        
        return combined
    
    def generate(
        self,
        prompt,
        sketch_path,
        num_inference_steps=20,
        guidance_scale=7.5,
        height=512,
        width=512,
        seed=None
    ):
        """Generate image from prompt and sketch"""
        
        if seed is not None:
            torch.manual_seed(seed)
            
        print(f"📝 Encoding prompt: '{prompt}'")
        text_embeddings = self.encode_prompt(prompt)
        
        print(f"🎨 Processing sketch...")
        sketch_tensor = self.preprocess_sketch(sketch_path)
        
        # Get sketch conditioning features
        with torch.no_grad():
            sketch_features = self.sketch_encoder(sketch_tensor)
        
        print(f"🔗 Combining text and sketch embeddings...")
        # Implement the missing combiner functionality
        combined_conditional = self.combine_embeddings(text_embeddings, sketch_features)
        
        # For unconditional, use empty text + zero sketch features
        uncond_text = self.encode_prompt("")
        uncond_sketch = torch.zeros_like(sketch_features)
        combined_unconditional = self.combine_embeddings(uncond_text, uncond_sketch)
        
        # Stack for classifier-free guidance
        encoder_hidden_states = torch.cat([combined_unconditional, combined_conditional])
        
        print(f"   Combined embeddings shape: {encoder_hidden_states.shape}")
        
        # Prepare latent space
        latents_shape = (1, 4, height // 8, width // 8)
        latents = torch.randn(
            latents_shape,
            device=self.device,
            dtype=self.dtype
        )
        
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma
        
        print(f"🔄 Generating...")
        
        # Denoising loop
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Generating")):
            
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample
            
            # Perform classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous noisy sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode latents to image
        print("🖼️ Decoding to image...")
        with torch.no_grad():
            latents = 1 / 0.18215 * latents
            image = self.vae.decode(latents).sample
        
        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        
        # Handle NaN values
        if np.isnan(image).any():
            print("⚠️ Warning: NaN values detected, replacing with zeros")
            image = np.nan_to_num(image, nan=0.0)
        
        image = (image * 255).astype(np.uint8)
        
        return Image.fromarray(image[0])

def main():
    """Test the fixed pipeline"""
    
    pipeline = FixedScribblePipeline(force_cpu=True)
    
    # Create test sketch
    test_sketch = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(test_sketch, (256, 256), 100, 255, 3)
    cv2.rectangle(test_sketch, (200, 350), (312, 400), 255, 2)
    test_sketch_pil = Image.fromarray(test_sketch)
    
    print("\n🧪 Running fixed ScribbleDiffusion test...")
    
    # Generate with sketch conditioning
    result = pipeline.generate(
        prompt="a beautiful red apple on a wooden table",
        sketch_path=test_sketch_pil,
        num_inference_steps=10,  # Quick for testing
        seed=42
    )
    
    # Save results
    test_sketch_pil.save("fixed_test_sketch.png")
    result.save("fixed_test_result.png")
    
    # Check result
    result_array = np.array(result)
    is_black = np.all(result_array < 10)
    
    if is_black:
        print("❌ Still producing black images")
        print(f"   Image range: [{result_array.min()}, {result_array.max()}]")
    else:
        print("✅ Fixed ScribbleDiffusion working!")
        print(f"   Image range: [{result_array.min()}, {result_array.max()}]")
        print(f"   Mean brightness: {result_array.mean():.1f}")
    
    print("📁 Results saved:")
    print("   - fixed_test_sketch.png")
    print("   - fixed_test_result.png")

if __name__ == "__main__":
    main()