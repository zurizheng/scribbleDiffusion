"""
Training script with cached edge detection for ScribbleDiffusion.
Uses pre-computed edge cache for fast training.
"""

import argparse
import logging
import os
import time
import torch
from pathlib import Path

from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm

from src.models.sketch_encoder import SketchCrossAttentionEncoder, SketchTextCombiner  
from src.training.losses import DiffusionLoss
from src.utils.config import load_config
from src.data.cached_coco_dataset import CachedCOCOScribbleDataset


logger = get_logger(__name__, log_level="INFO")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--rebuild_cache", action="store_true", help="Rebuild edge detection cache")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="tensorboard",
        project_dir=args.output_dir,
    )
    
    if accelerator.is_main_process:
        logging.basicConfig(
            format="%(message)s",  # Simplified format
            level=logging.WARNING,  # Only show warnings and errors
        )
    # Don't log accelerator state to avoid spam
    # logger.info(accelerator.state, main_process_only=False)
    
    # Set random seed
    if config.training.get("seed"):
        set_seed(config.training.seed)

    # Load pretrained models
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
    vae.requires_grad_(False)
    vae.eval()
    
    text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
    text_encoder.requires_grad_(False) 
    text_encoder.eval()
    
    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")

    # Initialize UNet
    unet = UNet2DConditionModel(
        sample_size=config.model.unet.sample_size,
        in_channels=config.model.unet.in_channels,
        out_channels=config.model.unet.out_channels,
        layers_per_block=config.model.unet.layers_per_block,
        block_out_channels=config.model.unet.block_out_channels,
        down_block_types=config.model.unet.down_block_types,
        up_block_types=config.model.unet.up_block_types,
        cross_attention_dim=config.model.unet.cross_attention_dim,
        attention_head_dim=config.model.unet.attention_head_dim,
        use_linear_projection=config.model.unet.use_linear_projection,
    )
    unet.requires_grad_(True)
    unet.train()

    # Initialize sketch encoder
    sketch_encoder = SketchCrossAttentionEncoder(
        in_channels=1,
        hidden_dim=512,
        num_sketch_tokens=77,
        cross_attention_dim=768,
    )
    sketch_encoder.requires_grad_(True)
    sketch_encoder.train()

    # Initialize sketch-text combiner
    sketch_text_combiner = SketchTextCombiner(
        cross_attention_dim=768,
        combination_method="concat",
    )
    sketch_text_combiner.requires_grad_(True)
    sketch_text_combiner.train()

    # Initialize noise scheduler
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=config.diffusion.num_train_timesteps,
        beta_schedule=config.diffusion.beta_schedule,
        prediction_type=config.diffusion.prediction_type,
        clip_sample=config.diffusion.clip_sample,
    )

    # Setup optimizer
    trainable_params = list(unet.parameters()) + list(sketch_encoder.parameters()) + list(sketch_text_combiner.parameters())
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        eps=config.training.epsilon,
    )

    # Setup dataset with cached edges
    train_dataset = CachedCOCOScribbleDataset(
        config=config.data,
        tokenizer=tokenizer,
        split="train",
        download=config.data.get("download_coco", True),
        rebuild_cache=args.rebuild_cache,
    )
    
    # Show cache information
    cache_info = train_dataset.get_cache_info()
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=2,  # Can use workers now since edges are cached
        pin_memory=True,
    )

    # Setup learning rate scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.training.max_train_steps,
        eta_min=config.training.learning_rate * 0.1,
    )

    # Setup loss function
    loss_fn = DiffusionLoss(config.training)

    # Prepare everything with accelerator
    unet, sketch_encoder, sketch_text_combiner, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, sketch_encoder, sketch_text_combiner, optimizer, train_dataloader, lr_scheduler
    )

    # Enable gradient checkpointing
    if hasattr(unet, 'enable_gradient_checkpointing'):
        unet.enable_gradient_checkpointing()

    # Move frozen models to device
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device, dtype=torch.float32)

    # Training loop with timing
    global_step = 0
    epoch = 0
    
    progress_bar = tqdm(
        range(config.training.max_train_steps),
        desc="Training",
        disable=not accelerator.is_main_process,
    )

    step_times = []
    start_time = time.time()

    while global_step < config.training.max_train_steps:
        
        for step, batch in enumerate(train_dataloader):
            step_start_time = time.time()
            
            with accelerator.accumulate(unet):
                
                # Get batch data
                images = batch["images"].to(accelerator.device)
                sketches = batch["sketches"].to(accelerator.device) 
                input_ids = batch["input_ids"].to(accelerator.device)

                # Encode images to latents
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Encode text
                with torch.no_grad():
                    text_embeddings = text_encoder(input_ids)[0]

                # Encode sketches
                sketch_embeddings = sketch_encoder(sketches)

                # Combine embeddings
                combined_embeddings = sketch_text_combiner(text_embeddings, sketch_embeddings)

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=accelerator.device
                ).long()

                # Add noise
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Forward pass
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=combined_embeddings
                ).sample

                # Calculate loss
                loss = loss_fn(model_pred, noise, timesteps)

                # Backward pass
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)

                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Clean up memory
                current_loss = loss.detach().cpu().item()
                del loss, model_pred, noise, latents, noisy_latents
                del images, sketches, text_embeddings, sketch_embeddings, combined_embeddings
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Timing
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            step_times.append(step_time)
            
            # Calculate statistics
            avg_step_time = sum(step_times) / len(step_times)
            recent_avg = sum(step_times[-10:]) / min(10, len(step_times))
            
            # Update progress
            if accelerator.is_main_process:
                memory_gb = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                
                progress_bar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'time': f'{step_time:.1f}s',
                    'mem': f'{memory_gb:.1f}GB'
                })
                progress_bar.update(1)
                
                # Log detailed info (suppress most output to avoid progress bar interference)
                if global_step % (config.logging.log_interval * 5) == 0:  # Log less frequently
                    total_time = time.time() - start_time
                    steps_per_sec = global_step / total_time if total_time > 0 else 0
                    eta_seconds = (config.training.max_train_steps - global_step) / steps_per_sec if steps_per_sec > 0 else 0
                    eta_minutes = eta_seconds / 60
                    
                    # Only log occasionally to avoid interfering with progress bar
                    pass

            # Save checkpoint at regular intervals
            if global_step % config.logging.save_interval == 0:
                if accelerator.is_main_process:
                    # Save checkpoint using accelerator's project directory
                    save_path = f"{args.output_dir}/{config.logging.project_name}/checkpoint-{global_step}"
                    os.makedirs(save_path, exist_ok=True)
                    
                    print(f"\n SAVING CHECKPOINT at step {global_step}")
                    print(f" Location: {save_path}")
                    
                    # Save model weights directly 
                    checkpoint_dict = {
                        'unet_state_dict': accelerator.get_state_dict(unet),
                        'sketch_encoder_state_dict': accelerator.get_state_dict(sketch_encoder),
                        'step': global_step,
                        'config': config.__dict__ if hasattr(config, '__dict__') else str(config)
                    }
                    model_path = f"{save_path}/pytorch_model.bin"
                    torch.save(checkpoint_dict, model_path)
                    print(f"✅ CHECKPOINT SAVED: {model_path}")

            global_step += 1
            
            if global_step >= config.training.max_train_steps:
                break

        epoch += 1

    # Final report
    if accelerator.is_main_process:
        # Save final checkpoint
        final_save_path = f"{args.output_dir}/{config.logging.project_name}/checkpoint-{global_step}"
        os.makedirs(final_save_path, exist_ok=True)
        
        print(f"\n🎯 SAVING FINAL CHECKPOINT at step {global_step}")
        print(f"📁 Location: {final_save_path}")
        
        # Save model weights directly
        final_checkpoint_dict = {
            'unet_state_dict': accelerator.get_state_dict(unet),
            'sketch_encoder_state_dict': accelerator.get_state_dict(sketch_encoder),
            'step': global_step,
            'config': config.__dict__ if hasattr(config, '__dict__') else str(config)
        }
        final_model_path = f"{final_save_path}/pytorch_model.bin"
        torch.save(final_checkpoint_dict, final_model_path)
        print(f"✅ FINAL CHECKPOINT SAVED: {final_model_path}")
        
        total_time = time.time() - start_time
        avg_step_time = sum(step_times) / len(step_times) if step_times else 0
        
        print(f"\nTraining complete!")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Total steps: {len(step_times)}")
        print(f"Average step time: {avg_step_time:.2f}s")
        if total_time > 0:
            print(f"Steps per second: {len(step_times)/total_time:.2f}")


if __name__ == "__main__":
    main()