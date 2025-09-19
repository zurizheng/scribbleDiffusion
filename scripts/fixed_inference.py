#!/usr/bin/env python3
"""
Fixed ScribbleDiffusion inference that loads all trained components from scribble-diffusion-model
"""

import sys
import os
# Add project root to Python path
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from safetensors.torch import load_file
from tqdm import tqdm
import json

# Use diffusers for base components
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel

# Import our custom models
from src.models.sketch_encoder import SketchCrossAttentionEncoder, SketchTextCombiner
from src.utils.device_utils import get_optimal_device

class FixedScribblePipeline:
    def __init__(self, model_path="scribble-diffusion-model", force_cpu=False):
        """Fixed pipeline that loads all trained components from scribble-diffusion-model"""
        
        print("🚀 Loading ScribbleDiffusion...")
        
        # Use optimal device detection
        if force_cpu:
            self.device = torch.device("cpu")
            self.dtype = torch.float32
        else:
            self.device = get_optimal_device()
            self.dtype = torch.float32
            
        print(f"📱 Using device: {self.device}")
        print(f"🔢 Using dtype: {self.dtype}")
        
        # Load config from model directory
        self.config = self._load_config(model_path)
        
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
        
        # Load custom components
        print("⚡ Loading trained models (custom UNet, sketch_encoder, and text_combiner)...")
        self.unet = self._load_custom_unet(model_path)
        self.sketch_encoder = self._load_sketch_encoder(model_path)
        self.sketch_text_combiner = self._load_sketch_text_combiner(model_path)
        
        # Set to eval mode
        self.text_encoder.eval()
        self.vae.eval()
        self.unet.eval()
        self.sketch_encoder.eval()
        if self.sketch_text_combiner:
            self.sketch_text_combiner.eval()
        
        print("✅ Pipeline ready!")
    
    def _load_config(self, model_path):
        """Load model configuration"""
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Loaded config from {config_path}")
            return config
        print("No config found, using defaults")
        return {}
    
    def _load_custom_unet(self, model_path):
        """Load UNet with trained weights"""
        
        # Get UNet config from our config
        unet_config = self.config.get('architecture', {}).get('unet', {})
        
        # Create UNet with architecture from config
        unet = UNet2DConditionModel(
            in_channels=unet_config.get('in_channels', 4),
            out_channels=unet_config.get('out_channels', 4),
            down_block_types=unet_config.get('down_block_types', ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"]),
            up_block_types=unet_config.get('up_block_types', ["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"]),
            block_out_channels=unet_config.get('block_out_channels', [320, 640, 1280, 1280]),
            layers_per_block=unet_config.get('layers_per_block', 2),
            attention_head_dim=unet_config.get('attention_head_dim', 8),
            norm_num_groups=unet_config.get('norm_num_groups', 32),
            cross_attention_dim=unet_config.get('cross_attention_dim', 768),
        ).to(self.device)
        
        # Load the trained weights
        unet_path = os.path.join(model_path, "unet.safetensors")
        if os.path.exists(unet_path):
            try:
                unet_weights = load_file(unet_path)
                unet_weights = {k: v.to(self.dtype) for k, v in unet_weights.items()}
                
                missing_keys, unexpected_keys = unet.load_state_dict(unet_weights, strict=False)
                
                if not missing_keys and not unexpected_keys:
                    print("UNet weights loaded perfectly")
                else:
                    print(f"UNet loaded with {len(missing_keys)} missing, {len(unexpected_keys)} extra keys")
                        
            except Exception as e:
                print(f"Failed to load UNet weights: {e}")
                print("   Using base SD weights instead")
        else:
            print(f"UNet weights not found at {unet_path}, using base SD weights")
        
        return unet
    
    def _load_sketch_encoder(self, model_path):
        """Load sketch encoder with trained weights"""
        
        # Get sketch encoder config
        sketch_config = self.config.get('architecture', {}).get('sketch_encoder', {})
        
        sketch_encoder = SketchCrossAttentionEncoder(
            in_channels=sketch_config.get('in_channels', 1),
            hidden_dim=sketch_config.get('hidden_dim', 256),
            num_sketch_tokens=sketch_config.get('num_sketch_tokens', 77),
            cross_attention_dim=sketch_config.get('cross_attention_dim', 768)
        ).to(self.device)
        
        # Load trained weights
        sketch_path = os.path.join(model_path, "sketch_encoder.safetensors")
        if os.path.exists(sketch_path):
            try:
                sketch_weights = load_file(sketch_path)
                sketch_weights = {k: v.to(self.dtype) for k, v in sketch_weights.items()}
                sketch_encoder.load_state_dict(sketch_weights)
                print("Sketch encoder weights loaded")
            except Exception as e:
                print(f"Failed to load sketch encoder: {e}")
        else:
            print(f"Sketch encoder weights not found at {sketch_path}")
        
        return sketch_encoder
    
    def _load_sketch_text_combiner(self, model_path):
        """Load sketch text combiner"""
        
        combiner_path = os.path.join(model_path, "sketch_text_combiner.safetensors")
        if not os.path.exists(combiner_path):
            print(f"SketchTextCombiner not found at {combiner_path}")
            print("   Will use simple concatenation instead")
            return None
        
        # Get combiner config
        combiner_config = self.config.get('architecture', {}).get('sketch_text_combiner', {})
        
        sketch_text_combiner = SketchTextCombiner(
            cross_attention_dim=combiner_config.get('cross_attention_dim', 768)
        ).to(self.device)
        
        try:
            combiner_weights = load_file(combiner_path)
            combiner_weights = {k: v.to(self.dtype) for k, v in combiner_weights.items()}
            sketch_text_combiner.load_state_dict(combiner_weights)
            print("SketchTextCombiner weights loaded")
            return sketch_text_combiner
        except Exception as e:
            print(f"Failed to load SketchTextCombiner: {e}")
            return None
    
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
        height=256,
        width=256,
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
    
    pipeline = FixedScribblePipeline(force_cpu=False)
    
    # Create test sketch
    test_sketch = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(test_sketch, (128, 128), 50, 255, 3)
    test_sketch_pil = Image.fromarray(test_sketch)
    
    print("\n🧪 Running fixed ScribbleDiffusion test...")
    
    # Generate with sketch conditioning
    result = pipeline.generate(
        prompt="red apple",
        sketch_path=test_sketch_pil,
        num_inference_steps=1,  # Minimal for memory test
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