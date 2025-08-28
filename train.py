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
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
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

from src.data.dataset import ScribbleDataset
from src.models.hint_encoder import HintEncoder
from src.training.losses import DiffusionLoss
from src.training.ema import EMAModel
from src.utils.validation import validate_model
from src.utils.logging import setup_logging

# Check minimum diffusers version
check_min_version("0.18.0")

logger = get_logger(__name__)


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
        cache_dir=config.paths.cache_dir,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        config.model.text_encoder.model_name,
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
        block_out_channels=(320, 640, 1280, 1280),
        layers_per_block=2,
        attention_head_dim=8,
        norm_num_groups=32,
        cross_attention_dim=768,
    )
    
    # Enable gradient checkpointing for memory efficiency
    if config.training.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    # Initialize HintEncoder with matching channels for future integration
    hint_encoder_config = dict(config.model.hint_encoder)
    # Use standard diffusers UNet channel progression: [320, 640, 1280, 1280]
    unet_channels = [320, 640, 1280, 1280]
    hint_encoder_config['unet_channels'] = unet_channels
    hint_encoder = HintEncoder(**hint_encoder_config)
    print(f"RTX 3090 configuration - hint injection channels: {unet_channels}")
    
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
        ema_hint_encoder = EMAModel(hint_encoder.parameters(), decay=config.training.ema_decay)
    
    # Setup optimizer
    trainable_params = list(unet.parameters()) + list(hint_encoder.parameters())
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
    
    # Check if using COCO dataset
    if config.data.get("dataset_type") == "coco" or config.data.dataset_name == "coco":
        from src.data.coco_dataset import COCOScribbleDataset
        train_dataset = COCOScribbleDataset(
            config=config.data,
            tokenizer=tokenizer,
            split="train",
            download=config.data.get("download_coco", True),
        )
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
    unet, hint_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, hint_encoder, optimizer, train_dataloader, lr_scheduler
    )
    
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
    
    for epoch in range(1000):  # Large number, we'll break based on max_train_steps
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Get batch data
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
                
                # Get hint features from sketch (for future ControlNet-style integration)
                hint_features = hint_encoder(sketch_conditioning)
                
                # Predict noise (standard UNet without hint injection for now)
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
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
            
            # Update EMA
            if accelerator.sync_gradients:
                if config.training.use_ema:
                    ema_unet.step(unet.parameters())
                    ema_hint_encoder.step(hint_encoder.parameters())
                global_step += 1
            
            # Logging
            if global_step % config.logging.log_interval == 0:
                logs = {
                    "train_loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                    "epoch": epoch,
                }
                accelerator.log(logs, step=global_step)
                logger.info(f"Step {global_step}: Loss = {loss.detach().item():.4f}")
            
            # Validation
            if global_step % config.validation.validation_steps == 0:
                logger.info("Running validation...")
                
                # Get the actual models for validation
                val_unet = unet  # Use original unet, not EMA for now
                val_hint_encoder = hint_encoder  # Use original hint_encoder, not EMA for now
                
                validation_logs = validate_model(
                    unet=val_unet,
                    hint_encoder=val_hint_encoder,
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    noise_scheduler=noise_scheduler,
                    config=config,
                    accelerator=accelerator,
                    global_step=global_step,
                )
                accelerator.log(validation_logs, step=global_step)
            
            # Save checkpoint
            if global_step % config.logging.save_interval == 0:
                if accelerator.is_main_process:
                    save_path = Path(config.paths.output_dir) / f"checkpoint-{global_step}"
                    accelerator.save_state(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")
            
            # Check if we've reached max steps
            if global_step >= config.training.max_train_steps:
                break
        
        if global_step >= config.training.max_train_steps:
            break
    
    # Final save
    if accelerator.is_main_process:
        final_save_path = Path(config.paths.output_dir) / "final_model"
        accelerator.save_state(final_save_path)
        logger.info(f"Training completed. Final model saved to {final_save_path}")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
