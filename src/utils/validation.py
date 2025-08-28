"""
Validation utilities for ScribbleDiffusion training.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
import io
import base64


def validate_model(
    unet,
    hint_encoder,
    vae,
    text_encoder,
    tokenizer,
    noise_scheduler,
    config,
    accelerator,
    global_step: int,
) -> Dict[str, Any]:
    """
    Run validation and generate sample images.
    
    Returns validation metrics and sample images.
    """
    device = accelerator.device
    
    # Set models to eval mode
    unet.eval()
    hint_encoder.eval()
    
    # Prepare validation prompts and sketches
    val_prompts = [
        "a red rose",
        "a blue car",
        "a yellow house",
        "a green tree",
    ]
    
    # Create simple validation sketches (you'd load real ones in practice)
    val_sketches = []
    for i in range(len(val_prompts)):
        # Create a simple sketch (circle, square, etc.)
        sketch = torch.zeros(1, 1, 256, 256)
        # Add some simple lines
        sketch[0, 0, 50:200, 100:102] = 1.0  # Vertical line
        sketch[0, 0, 100:102, 50:200] = 1.0  # Horizontal line
        val_sketches.append(sketch.to(device))
    
    generated_images = []
    
    with torch.no_grad():
        for prompt, sketch in zip(val_prompts, val_sketches):
            # Tokenize prompt
            text_inputs = tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=77,
                return_tensors="pt",
            )
            text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
            
            # Get hint features
            hint_features = hint_encoder(sketch)
            
            # Generate image
            image = generate_image(
                unet=unet,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                noise_scheduler=noise_scheduler,
                text_embeddings=text_embeddings,
                hint_features=hint_features,
                config=config,
                device=device,
            )
            
            generated_images.append(image)
    
    # Convert images to format suitable for logging
    image_grid = create_image_grid(generated_images, val_prompts)
    
    # Calculate some basic metrics (placeholder)
    metrics = {
        "val_samples": len(generated_images),
        "val_step": global_step,
    }
    
    # Add image to metrics for logging
    if accelerator.is_main_process:
        metrics["validation_images"] = image_grid
    
    # Set models back to train mode
    unet.train()
    hint_encoder.train()
    
    return metrics


def generate_image(
    unet,
    vae,
    text_encoder,
    tokenizer,
    noise_scheduler,
    text_embeddings: torch.Tensor,
    hint_features: Dict[str, torch.Tensor],
    config,
    device,
    num_inference_steps: int = 50,
) -> Image.Image:
    """Generate a single image using the diffusion model."""
    
    # Initialize random latents
    latents = torch.randn(
        (1, 4, 64, 64),  # VAE latent size for 512x512 image
        device=device,
        dtype=torch.float32,
    )
    
    # Set timesteps
    noise_scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = noise_scheduler.timesteps
    
    # Guidance scales
    guidance_scale_text = config.validation.get("guidance_scale_text", 7.5)
    guidance_scale_sketch = config.validation.get("guidance_scale_sketch", 1.5)
    
    # Prepare unconditional embeddings for CFG
    uncond_tokens = tokenizer(
        "",
        truncation=True,
        padding="max_length",
        max_length=77,
        return_tensors="pt",
    )
    uncond_embeddings = text_encoder(uncond_tokens.input_ids.to(device))[0]
    
    # Denoising loop
    for i, t in enumerate(timesteps):
        # Expand latents for CFG (unconditional + text conditional)
        latent_model_input = torch.cat([latents] * 2)  # uncond, text
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
        
        # Prepare embeddings for CFG
        encoder_hidden_states = torch.cat([
            uncond_embeddings,      # Unconditional
            text_embeddings,        # Text conditional
        ])
        
        # Predict noise (standard UNet without hint injection for now)
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
        ).sample
        
        # Perform CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale_text * (noise_pred_text - noise_pred_uncond)
        
        # Split predictions
        
        # Compute previous sample
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
    
    # Decode latents to image
    latents = latents / vae.config.scaling_factor
    images = vae.decode(latents).sample
    
    # Convert to PIL Image
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype("uint8")
    
    image = Image.fromarray(images[0])
    return image


def create_image_grid(images, prompts):
    """Create a grid of images with captions for logging."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    for i, (image, prompt) in enumerate(zip(images, prompts)):
        if i < len(axes):
            axes[i].imshow(image)
            axes[i].set_title(prompt, fontsize=12)
            axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Convert to image for logging
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)
