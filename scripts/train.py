"""
Main training script for ScribbleDiffusion model.
"""

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.resolve()  # Go up one more level since we're in scripts/
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from omegaconf import OmegaConf
from safetensors.torch import save_file

# Import our models and utilities
from src.models.unet import SketchConditionedUNet
from src.models.hint_encoder import HintEncoder  
from src.training.losses import DiffusionLoss
from src.utils.device_utils import clear_device_cache, synchronize_device, set_memory_fraction, get_device_memory_gb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import DDIMScheduler, AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from src.training.losses import DiffusionLoss
from src.training.ema import EMAModel
from src.utils.validation import validate_model
from src.utils.logging import setup_logging

# Check minimum diffusers version
check_min_version("0.18.0")

logger = get_logger(__name__)


def save_model_checkpoint(unet, sketch_encoder, sketch_text_combiner, output_dir, step, accelerator):
    """Save all model components properly"""
    
    save_dir = Path(output_dir) / f"checkpoint-{step}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unwrapped models from accelerator
    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_sketch_encoder = accelerator.unwrap_model(sketch_encoder)
    unwrapped_sketch_text_combiner = accelerator.unwrap_model(sketch_text_combiner)
    
    # Save UNet
    unet_state = unwrapped_unet.state_dict()
    unet_path = save_dir / "unet.safetensors"
    save_file(unet_state, unet_path)
    
    # Save Sketch Encoder
    sketch_encoder_state = unwrapped_sketch_encoder.state_dict()
    sketch_encoder_path = save_dir / "sketch_encoder.safetensors"
    save_file(sketch_encoder_state, sketch_encoder_path)
    
    # Save Sketch Text Combiner (THE MISSING PIECE!)
    combiner_state = unwrapped_sketch_text_combiner.state_dict()
    combiner_path = save_dir / "sketch_text_combiner.safetensors"
    save_file(combiner_state, combiner_path)
    
    # Save training info
    training_info = {
        "step": step,
        "model_components": ["unet", "sketch_encoder", "sketch_text_combiner"]
    }
    
    import json
    with open(save_dir / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    logger.info(f"✅ Saved complete model checkpoint to {save_dir}")
    logger.info(f"   - UNet: {unet_path}")
    logger.info(f"   - Sketch Encoder: {sketch_encoder_path}")
    logger.info(f"   - Sketch Text Combiner: {combiner_path}")
    
    return save_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Train ScribbleDiffusion model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    if args.output_dir:
        config.paths.output_dir = args.output_dir
    
    # Setup logging and accelerator
    logging_dir = Path(config.paths.output_dir) / config.paths.logging_dir
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.paths.output_dir,
        logging_dir=logging_dir,
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=config.logging.log_with,
        project_config=accelerator_project_config,
    )
    
    # Setup logging
    setup_logging(accelerator)
    
    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    logger.info(f"Training configuration:\n{OmegaConf.to_yaml(config)}")
    
    # Load pretrained models (frozen)
    logger.info("Loading pretrained models...")
    vae = AutoencoderKL.from_pretrained(
        config.model.vae.model_name,
        subfolder=config.model.vae.subfolder,
        cache_dir=config.paths.cache_dir,
    )
    vae.requires_grad_(False)
    
    tokenizer = CLIPTokenizer.from_pretrained(
        config.model.text_encoder.model_name,
        subfolder="tokenizer",
        cache_dir=config.paths.cache_dir,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        config.model.text_encoder.model_name,
        subfolder=config.model.text_encoder.subfolder,
        cache_dir=config.paths.cache_dir,
    )
    text_encoder.requires_grad_(False)
    
    # Initialize trainable models
    logger.info("Initializing trainable models...")
    
    # Use standard diffusers UNet2DConditionModel for compatibility
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel(
        in_channels=config.model.unet.in_channels,
        out_channels=config.model.unet.out_channels,
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        block_out_channels=(320, 640, 1280, 1280),  # Keep standard SD 1.5 channels
        layers_per_block=2,
        attention_head_dim=8,
        norm_num_groups=32,
        cross_attention_dim=768,
    )
    
    # Enable gradient checkpointing for memory efficiency
    if config.training.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    # Initialize HintEncoder with cross-attention approach
    from src.models.sketch_encoder import SketchCrossAttentionEncoder, SketchTextCombiner
    
    sketch_encoder = SketchCrossAttentionEncoder(
        in_channels=1,
        hidden_dim=256,  # Reduced from 512 to save memory
        num_sketch_tokens=77,  # Match text sequence length
        cross_attention_dim=768,  # Match CLIP embedding dimension
    )
    
    sketch_text_combiner = SketchTextCombiner(
        cross_attention_dim=768,
        combination_method="concat",  # Start with simple concatenation
    )
    
    print(f"RTX 3090 configuration - using cross-attention sketch conditioning")
    
    # Initialize noise scheduler
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=config.diffusion.num_train_timesteps,
        beta_schedule=config.diffusion.beta_schedule,
        prediction_type=config.diffusion.prediction_type,
        clip_sample=config.diffusion.clip_sample,
    )
    
    # Initialize EMA
    if config.training.use_ema:
        ema_unet = EMAModel(unet.parameters(), decay=config.training.ema_decay)
        ema_sketch_encoder = EMAModel(sketch_encoder.parameters(), decay=config.training.ema_decay)
        ema_sketch_text_combiner = EMAModel(sketch_text_combiner.parameters(), decay=config.training.ema_decay)
    
    # Setup optimizer
    trainable_params = (
        list(unet.parameters()) + 
        list(sketch_encoder.parameters()) + 
        list(sketch_text_combiner.parameters())
    )
    if config.training.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.training.learning_rate,
            betas=(config.training.beta1, config.training.beta2),
            weight_decay=config.training.weight_decay,
            eps=config.training.epsilon,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.training.optimizer}")
    
    # Setup dataset and dataloader
    logger.info("Setting up dataset...")
    
    # Check dataset type
    dataset_type = config.data.get("dataset_type", "default")
    
    if dataset_type == "coco" or config.data.get("dataset_name") == "coco":
        from src.data.coco_dataset import COCOScribbleDataset
        train_dataset = COCOScribbleDataset(
            config=config.data,
            tokenizer=tokenizer,
            split="train",
            download=config.data.get("download_coco", True),
        )
    elif dataset_type == "fruit":
        from src.data.fruit_dataset import FruitDataset
        train_dataset = FruitDataset(
            data_dir=config.data.data_dir,
            image_size=config.data.get("image_size", 256),
            create_sketches=True
        )
        logger.info(f"✅ Loaded fruit dataset: {len(train_dataset)} images")
    else:
        from src.data.dataset import ScribbleDataset
        train_dataset = ScribbleDataset(
            config=config.data,
            tokenizer=tokenizer,
            split="train",
        )
    
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.training.batch_size,
        num_workers=2,  # Conservative multiprocessing
        pin_memory=True,
    )
    
    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        config.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.training.num_warmup_steps * accelerator.num_processes,
        num_training_steps=config.training.max_train_steps * accelerator.num_processes,
    )
    
    # Setup loss function
    loss_fn = DiffusionLoss(config.training)
    
    # Prepare everything with accelerator
    unet, sketch_encoder, sketch_text_combiner, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, sketch_encoder, sketch_text_combiner, optimizer, train_dataloader, lr_scheduler
    )
    
    # Enable gradient checkpointing for memory savings
    if hasattr(unet, 'enable_gradient_checkpointing'):
        unet.enable_gradient_checkpointing()
        logger.info("Enabled gradient checkpointing for UNet")
    
    if hasattr(sketch_encoder, 'enable_gradient_checkpointing'):
        sketch_encoder.enable_gradient_checkpointing()
        logger.info("Enabled gradient checkpointing for SketchEncoder")
    
    # Move frozen models to device
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device, dtype=torch.float32)
    
    # Initialize tracking
    if accelerator.is_main_process:
        run_name = config.logging.run_name or f"scribble-diffusion-{int(time.time())}"
        
        # Create a flattened config with only basic types for TensorBoard
        tracking_config = {
            "learning_rate": float(config.training.learning_rate),
            "batch_size": int(config.training.batch_size),
            "max_train_steps": int(config.training.max_train_steps),
            "gradient_accumulation_steps": int(config.training.gradient_accumulation_steps),
            "image_size": int(config.data.image_size),
            "dataset_type": str(config.data.dataset_type),
            "unet_channels": int(config.model.unet.model_channels),
            "mixed_precision": str(config.training.mixed_precision),
        }
        
        accelerator.init_trackers(
            config.logging.project_name,
            config=tracking_config,
            init_kwargs={"wandb": {"name": run_name}},
        )
    
    # Training loop
    logger.info("Starting training...")
    global_step = 0
    
    # Set device memory management for better fragmentation handling
    clear_device_cache()
    set_memory_fraction(0.95)
    
    # Create progress bar for total steps
    if accelerator.is_main_process:
        progress_bar = tqdm(
            total=config.training.max_train_steps,
            desc="Training",
            unit="step",
            ncols=120,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )
        # Initialize loss tracking
        recent_losses = []
        max_recent_losses = 100  # Track last 100 losses for smoothing
    
    for epoch in range(1000):  # Large number, we'll break based on max_train_steps
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Handle different dataset formats
                if isinstance(batch, dict) and "image" in batch:
                    # Fruit dataset format
                    images = batch["image"]
                    sketches = batch["sketch"] 
                    
                    # Convert text prompts to tokens
                    text_prompts = batch["text_prompt"]
                    input_ids = tokenizer(
                        text_prompts,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).input_ids.to(accelerator.device)
                else:
                    # Original dataset format
                    images = batch["images"]
                    sketches = batch["sketches"]
                    input_ids = batch["input_ids"]
                
                # Encode images to latents
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device
                ).long()
                
                # Add noise to latents
                noise = torch.randn_like(latents)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]
                
                # Apply classifier-free guidance conditioning drops
                if random.random() < config.training.text_drop_prob:
                    encoder_hidden_states = torch.zeros_like(encoder_hidden_states)
                
                sketch_conditioning = sketches
                if random.random() < config.training.sketch_drop_prob:
                    sketch_conditioning = torch.zeros_like(sketches)
                
                # Get sketch embeddings
                sketch_embeddings = sketch_encoder(sketch_conditioning)
                
                # Combine text and sketch embeddings
                combined_embeddings = sketch_text_combiner(encoder_hidden_states, sketch_embeddings)
                
                # Predict noise with combined conditioning
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=combined_embeddings,
                ).sample
                
                # Calculate loss
                loss = loss_fn(model_pred, noise, timesteps)
                
                # Backward pass
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Aggressive memory cleanup
                del loss  # Delete loss tensor immediately
                del model_pred, noise  # Delete intermediate tensors
                if global_step % 10 == 0:
                    clear_device_cache()
                    synchronize_device()  # Ensure cleanup completes
            
            # Update EMA and global step
            if accelerator.sync_gradients:
                if config.training.use_ema:
                    ema_unet.step(unet.parameters())
                    ema_sketch_encoder.step(sketch_encoder.parameters())
                    ema_sketch_text_combiner.step(sketch_text_combiner.parameters())
                global_step += 1
                
                # Update progress bar (detach and move to CPU to avoid GPU accumulation)
                if accelerator.is_main_process:
                    current_loss = loss.detach().cpu().item()  # Move to CPU immediately
                    recent_losses.append(current_loss)
                    if len(recent_losses) > max_recent_losses:
                        recent_losses.pop(0)
                    
                    avg_loss = sum(recent_losses) / len(recent_losses)
                    lr = lr_scheduler.get_last_lr()[0]
                    
                    progress_bar.set_postfix({
                        'loss': f'{current_loss:.4f}',
                        'avg_loss': f'{avg_loss:.4f}',
                        'lr': f'{lr:.2e}',
                        'epoch': epoch
                    })
                    progress_bar.update(1)
            
            # Logging (only on gradient sync steps to avoid spam)
            if accelerator.sync_gradients and global_step % config.logging.log_interval == 0:
                current_loss = loss.detach().cpu().item()  # Move to CPU immediately
                logs = {
                    "train_loss": current_loss,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                    "epoch": epoch,
                }
                accelerator.log(logs, step=global_step)
                # Don't duplicate the loss logging since progress bar shows it
                if global_step % (config.logging.log_interval * 5) == 0:  # Less frequent console logs
                    logger.info(f"Step {global_step}: Loss = {current_loss:.4f}")
            
            # Validation (temporarily disabled for Phase 2)
            if False and global_step % config.validation.validation_steps == 0:
                logger.info("Running validation...")
                
                # Get the actual models for validation
                val_unet = unet  # Use original unet, not EMA for now
                val_sketch_encoder = sketch_encoder  # Use original sketch_encoder, not EMA for now
                val_sketch_text_combiner = sketch_text_combiner  # Use original combiner, not EMA for now
                
                validation_logs = validate_model(
                    unet=val_unet,
                    sketch_encoder=val_sketch_encoder,
                    sketch_text_combiner=val_sketch_text_combiner,
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    noise_scheduler=noise_scheduler,
                    config=config,
                    accelerator=accelerator,
                    global_step=global_step,
                )
                accelerator.log(validation_logs, step=global_step)
            
            # Save checkpoint (skip step 0 to avoid spam)
            if global_step > 0 and global_step % config.logging.save_interval == 0:
                if accelerator.is_main_process:
                    # Save proper model checkpoint with all components
                    save_model_checkpoint(
                        unet, sketch_encoder, sketch_text_combiner,
                        config.paths.output_dir, global_step, accelerator
                    )
                    
                    # Also save accelerator state for resuming
                    accelerator_save_path = Path(config.paths.output_dir) / f"accelerator-{global_step}"
                    accelerator.save_state(accelerator_save_path)
            
            # Check if we've reached max steps
            if global_step >= config.training.max_train_steps:
                if accelerator.is_main_process:
                    progress_bar.close()
                break
        
        if global_step >= config.training.max_train_steps:
            break
    
    # Final save
    if accelerator.is_main_process:
        if 'progress_bar' in locals():
            progress_bar.close()
        
        # Save final complete model
        final_save_path = save_model_checkpoint(
            unet, sketch_encoder, sketch_text_combiner,
            config.paths.output_dir, global_step, accelerator
        )
        
        # Also save final accelerator state
        final_accelerator_path = Path(config.paths.output_dir) / "final_accelerator"
        accelerator.save_state(final_accelerator_path)
        
        logger.info(f"🎉 Training completed!")
        logger.info(f"📁 Final model saved to: {final_save_path}")
        logger.info(f"📁 All components saved: UNet, SketchEncoder, SketchTextCombiner")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
