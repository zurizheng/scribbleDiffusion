"""
Complete inference pipeline for ScribbleDiffusion.
Handles end-to-end generation with attention tracking.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

from diffusers import DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer


@dataclass
class ScribbleDiffusionOutput:
    """Output class for ScribbleDiffusion pipeline."""
    images: List[Image.Image]
    attention_maps: Optional[Dict[str, torch.Tensor]] = None
    timeline_images: Optional[List[Image.Image]] = None


class ScribbleDiffusionPipeline:
    """
    Complete pipeline for sketch + text to image generation.
    
    Supports:
    - Dual guidance (text + sketch)
    - Attention map extraction
    - Timeline visualization
    - Flexible inference parameters
    """
    
    def __init__(
        self,
        unet,
        hint_encoder,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        scheduler: DDIMScheduler,
        device: str = "cuda",
    ):
        self.unet = unet
        self.hint_encoder = hint_encoder
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.device = device
        
        # Move models to device
        self.unet.to(device)
        self.hint_encoder.to(device)
        self.vae.to(device)
        self.text_encoder.to(device)
        
        # Set to eval mode
        self.unet.eval()
        self.hint_encoder.eval()
        self.vae.eval()
        self.text_encoder.eval()
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load pipeline from saved checkpoint."""
        # This is a placeholder for loading your trained model
        # You'll implement this based on your checkpoint format
        
        # Load pretrained components
        vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="vae",
        )
        
        text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        
        tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        
        scheduler = DDIMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="scheduler",
        )
        
        # Load your trained models
        # unet = load_trained_unet(model_path)
        # hint_encoder = load_trained_hint_encoder(model_path)
        
        # For now, create dummy models (you'll replace this)
        from ..models.unet import SketchConditionedUNet
        from ..models.hint_encoder import HintEncoder
        
        unet = SketchConditionedUNet()
        hint_encoder = HintEncoder()
        
        return cls(
            unet=unet,
            hint_encoder=hint_encoder,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            **kwargs
        )
    
    def encode_text(self, prompt: str) -> torch.Tensor:
        """Encode text prompt to embeddings."""
        text_inputs = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=77,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_inputs.input_ids.to(self.device)
            )[0]
        
        return text_embeddings
    
    def prepare_sketch(self, sketch: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """Prepare sketch input for hint encoder."""
        if isinstance(sketch, Image.Image):
            sketch = np.array(sketch)
        
        # Convert to grayscale if needed
        if len(sketch.shape) == 3:
            sketch = sketch.mean(axis=2)
        
        # Normalize to [0, 1]
        sketch = sketch.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        sketch_tensor = torch.from_numpy(sketch).unsqueeze(0).unsqueeze(0)
        
        return sketch_tensor.to(self.device)
    
    def __call__(
        self,
        prompt: str,
        sketch: Union[np.ndarray, Image.Image],
        guidance_scale_text: float = 7.5,
        guidance_scale_sketch: float = 1.5,
        num_inference_steps: int = 50,
        height: int = 256,
        width: int = 256,
        generator: Optional[torch.Generator] = None,
        return_attention_maps: bool = False,
        return_timeline: bool = False,
        timeline_steps: List[int] = None,
    ) -> ScribbleDiffusionOutput:
        """
        Generate image from sketch and text prompt.
        
        Args:
            prompt: Text description
            sketch: Input sketch (image or array)
            guidance_scale_text: CFG scale for text
            guidance_scale_sketch: CFG scale for sketch
            num_inference_steps: Number of denoising steps
            height: Output image height
            width: Output image width
            generator: Random generator for reproducibility
            return_attention_maps: Whether to return attention maps
            return_timeline: Whether to return timeline images
            timeline_steps: Specific steps to save for timeline
            
        Returns:
            ScribbleDiffusionOutput with generated images and optional visualizations
        """
        
        # Set timeline steps if not provided
        if timeline_steps is None:
            timeline_steps = [
                num_inference_steps - 1,  # Initial noise
                3 * num_inference_steps // 4,  # 75% done
                num_inference_steps // 2,  # 50% done
                num_inference_steps // 4,  # 25% done
                0,  # Final result
            ]
        
        # Prepare inputs
        text_embeddings = self.encode_text(prompt)
        sketch_tensor = self.prepare_sketch(sketch)
        
        # Get hint features
        with torch.no_grad():
            hint_features = self.hint_encoder(sketch_tensor)
        
        # Prepare unconditional embeddings for CFG
        uncond_embeddings = self.encode_text("")
        
        # Initialize latents
        shape = (1, 4, height // 8, width // 8)  # VAE latent size
        latents = torch.randn(shape, generator=generator, device=self.device)
        
        # Set scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # Storage for timeline and attention maps
        timeline_images = []
        attention_maps = {}
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 3)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Prepare conditioning
            encoder_hidden_states = torch.cat([
                uncond_embeddings,      # Unconditional
                text_embeddings,        # Text only
                text_embeddings,        # Text + sketch
            ])
            
            # Prepare hint features for CFG
            batch_hint_features = {}
            for res, features in hint_features.items():
                batch_hint_features[res] = torch.cat([
                    torch.zeros_like(features),  # No sketch
                    torch.zeros_like(features),  # No sketch
                    features,                    # With sketch
                ])
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    hint_features=batch_hint_features,
                ).sample
            
            # Extract attention maps if requested
            if return_attention_maps and hasattr(self.unet, 'attention_maps'):
                # Store attention maps for this timestep
                attention_maps[f"step_{i}"] = self.unet.attention_maps
            
            # Split predictions for CFG
            noise_pred_uncond, noise_pred_text, noise_pred_sketch = noise_pred.chunk(3)
            
            # Apply dual classifier-free guidance
            noise_pred = (
                noise_pred_uncond +
                guidance_scale_text * (noise_pred_text - noise_pred_uncond) +
                guidance_scale_sketch * (noise_pred_sketch - noise_pred_text)
            )
            
            # Compute previous sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            # Save timeline images if requested
            if return_timeline and i in timeline_steps:
                timeline_img = self.decode_latents(latents.clone())
                timeline_images.append(timeline_img)
        
        # Decode final latents
        final_image = self.decode_latents(latents)
        
        return ScribbleDiffusionOutput(
            images=[final_image],
            attention_maps=attention_maps if return_attention_maps else None,
            timeline_images=timeline_images if return_timeline else None,
        )
    
    def decode_latents(self, latents: torch.Tensor) -> Image.Image:
        """Decode latents to PIL image."""
        latents = latents / self.vae.config.scaling_factor
        
        with torch.no_grad():
            images = self.vae.decode(latents).sample
        
        # Convert to PIL
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")
        
        return Image.fromarray(images[0])
