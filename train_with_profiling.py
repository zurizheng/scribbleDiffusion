"""
Memory Profiled Training Script for ScribbleDiffusion
Tracks GPU memory usage at each step to identify bottlenecks
"""

import argparse
import logging
import os
import torch
import gc
from pathlib import Path

# Import our memory profiler
from src.utils.memory_profiler import GPUMemoryProfiler, profile_model_memory, MemoryProfileContext, emergency_memory_cleanup

# Original imports
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm

from src.models.sketch_encoder import SketchCrossAttentionEncoder, SketchTextCombiner  
from src.training.losses import DiffusionLoss
from src.training.ema import EMAModel
from src.utils.config import load_config


logger = get_logger(__name__, log_level="INFO")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--profile_memory", action="store_true", help="Enable detailed memory profiling")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize memory profiler
    profiler = GPUMemoryProfiler(enabled=args.profile_memory or True)  # Always enable for debugging
    profiler.profile_memory("SCRIPT_START", "Beginning of training script")

    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="tensorboard",
        project_dir=args.output_dir,
    )
    
    if accelerator.is_main_process:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
    logger.info(accelerator.state, main_process_only=False)
    
    # Set random seed
    if config.training.get("seed"):
        set_seed(config.training.seed)
    
    profiler.profile_memory("ACCELERATOR_INIT", "After accelerator initialization")

    # Load pretrained models
    logger.info("Loading pretrained models...")
    
    with MemoryProfileContext(profiler, "LOAD_VAE"):
        vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
        vae.requires_grad_(False)
        vae.eval()
    
    with MemoryProfileContext(profiler, "LOAD_TEXT_ENCODER"):
        text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
        text_encoder.requires_grad_(False) 
        text_encoder.eval()
    
    with MemoryProfileContext(profiler, "LOAD_TOKENIZER"):
        tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")

    # Initialize UNet
    logger.info("Initializing UNet...")
    with MemoryProfileContext(profiler, "INIT_UNET"):
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

    profile_model_memory(unet, "UNet", profiler)

    # Initialize sketch encoder
    logger.info("Initializing sketch encoder...")
    with MemoryProfileContext(profiler, "INIT_SKETCH_ENCODER"):
        sketch_encoder = SketchCrossAttentionEncoder(
            in_channels=1,
            hidden_dim=256,  # Reduced for memory
            num_sketch_tokens=77,
            cross_attention_dim=768,
        )
        sketch_encoder.requires_grad_(True)
        sketch_encoder.train()

    profile_model_memory(sketch_encoder, "SketchEncoder", profiler)

    # Initialize sketch-text combiner
    with MemoryProfileContext(profiler, "INIT_SKETCH_COMBINER"):
        sketch_text_combiner = SketchTextCombiner(
            cross_attention_dim=768,
            combination_method="concat",
        )
        sketch_text_combiner.requires_grad_(True)
        sketch_text_combiner.train()

    profile_model_memory(sketch_text_combiner, "SketchCombiner", profiler)

    # Initialize noise scheduler
    with MemoryProfileContext(profiler, "INIT_SCHEDULER"):
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=config.diffusion.num_train_timesteps,
            beta_schedule=config.diffusion.beta_schedule,
            prediction_type=config.diffusion.prediction_type,
            clip_sample=config.diffusion.clip_sample,
        )

    # Setup optimizer
    logger.info("Setting up optimizer...")
    trainable_params = list(unet.parameters()) + list(sketch_encoder.parameters()) + list(sketch_text_combiner.parameters())
    
    with MemoryProfileContext(profiler, "INIT_OPTIMIZER"):
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            eps=config.training.epsilon,
        )

    # Setup dataset
    logger.info("Setting up dataset...")
    with MemoryProfileContext(profiler, "INIT_DATASET"):
        from src.data.coco_dataset import COCOScribbleDataset
        train_dataset = COCOScribbleDataset(
            config=config.data,
            tokenizer=tokenizer,
            split="train",
            download=config.data.get("download_coco", True),
        )
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=0,  # Reduced for memory
            pin_memory=False,  # Disabled for memory
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
    logger.info("Preparing models with accelerator...")
    with MemoryProfileContext(profiler, "ACCELERATOR_PREPARE"):
        unet, sketch_encoder, sketch_text_combiner, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, sketch_encoder, sketch_text_combiner, optimizer, train_dataloader, lr_scheduler
        )

    # Enable gradient checkpointing
    if hasattr(unet, 'enable_gradient_checkpointing'):
        unet.enable_gradient_checkpointing()
        profiler.profile_memory("UNET_GRAD_CHECKPOINT", "Enabled UNet gradient checkpointing")

    # Move frozen models to device
    with MemoryProfileContext(profiler, "MOVE_FROZEN_MODELS"):
        vae.to(accelerator.device, dtype=torch.float32)
        text_encoder.to(accelerator.device, dtype=torch.float32)

    profiler.profile_memory("TRAINING_SETUP_COMPLETE", "All models loaded and ready for training")

    # Training loop
    logger.info("Starting training...")
    global_step = 0
    epoch = 0
    
    try:
        progress_bar = tqdm(
            range(config.training.max_train_steps),
            desc="Training",
            disable=not accelerator.is_main_process,
        )

        while global_step < config.training.max_train_steps:
            profiler.profile_memory(f"EPOCH_{epoch}_START", f"Beginning of epoch {epoch}")
            
            for step, batch in enumerate(train_dataloader):
                profiler.step()
                
                with MemoryProfileContext(profiler, f"BATCH_{step}"):
                    with accelerator.accumulate(unet):
                        
                        # Get batch data
                        with MemoryProfileContext(profiler, "GET_BATCH_DATA"):
                            images = batch["images"].to(accelerator.device)
                            sketches = batch["sketches"].to(accelerator.device) 
                            input_ids = batch["input_ids"].to(accelerator.device)
                            
                            profiler.profile_memory("BATCH_TO_DEVICE", f"Batch size: {images.shape[0]}, Image shape: {images.shape}")

                        # Encode images to latents
                        with MemoryProfileContext(profiler, "VAE_ENCODE"):
                            with torch.no_grad():
                                latents = vae.encode(images).latent_dist.sample()
                                latents = latents * vae.config.scaling_factor

                        # Encode text
                        with MemoryProfileContext(profiler, "TEXT_ENCODE"):
                            with torch.no_grad():
                                text_embeddings = text_encoder(input_ids)[0]

                        # Encode sketches
                        with MemoryProfileContext(profiler, "SKETCH_ENCODE"):
                            sketch_embeddings = sketch_encoder(sketches)

                        # Combine embeddings
                        with MemoryProfileContext(profiler, "COMBINE_EMBEDDINGS"):
                            combined_embeddings = sketch_text_combiner(text_embeddings, sketch_embeddings)

                        # Sample noise and timesteps
                        with MemoryProfileContext(profiler, "SAMPLE_NOISE"):
                            noise = torch.randn_like(latents)
                            timesteps = torch.randint(
                                0, noise_scheduler.config.num_train_timesteps,
                                (latents.shape[0],), device=accelerator.device
                            ).long()

                        # Add noise
                        with MemoryProfileContext(profiler, "ADD_NOISE"):
                            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                        # Forward pass
                        with MemoryProfileContext(profiler, "UNET_FORWARD"):
                            model_pred = unet(
                                noisy_latents,
                                timesteps,
                                encoder_hidden_states=combined_embeddings
                            ).sample

                        # Calculate loss
                        with MemoryProfileContext(profiler, "CALCULATE_LOSS"):
                            loss = loss_fn(model_pred, noise, timesteps)

                        # Backward pass
                        with MemoryProfileContext(profiler, "BACKWARD_PASS"):
                            accelerator.backward(loss)
                            
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(trainable_params, 1.0)

                        # Optimizer step
                        with MemoryProfileContext(profiler, "OPTIMIZER_STEP"):
                            optimizer.step()
                            lr_scheduler.step()
                            optimizer.zero_grad()

                        # Memory cleanup
                        with MemoryProfileContext(profiler, "MEMORY_CLEANUP"):
                            current_loss = loss.detach().cpu().item()
                            del loss, model_pred, noise, latents, noisy_latents
                            del images, sketches, text_embeddings, sketch_embeddings, combined_embeddings
                            
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()

                        profiler.profile_memory(f"STEP_{global_step}_COMPLETE", f"Loss: {current_loss:.4f}")

                        # Update progress
                        if accelerator.is_main_process:
                            progress_bar.set_postfix({
                                'loss': f'{current_loss:.4f}',
                                'memory_gb': f'{torch.cuda.memory_allocated() / 1024**3:.2f}' if torch.cuda.is_available() else '0',
                            })
                            progress_bar.update(1)

                        global_step += 1
                        
                        # Check if we should stop
                        if global_step >= config.training.max_train_steps:
                            break

            epoch += 1

    except torch.cuda.OutOfMemoryError as e:
        profiler.profile_memory("OOM_ERROR", f"CUDA OOM at step {global_step}")
        profiler.print_memory_breakdown()
        emergency_memory_cleanup()
        
        # Save profile data for analysis
        profile_path = os.path.join(args.output_dir, "memory_profile_oom.json")
        profiler.save_profile(profile_path)
        
        print(f"\n🚨 CUDA Out of Memory Error at step {global_step}")
        print(f"Peak memory usage: {profiler.get_peak_memory():.2f}GB")
        print(f"Memory profile saved to: {profile_path}")
        
        raise e

    except Exception as e:
        profiler.profile_memory("ERROR", f"Training error: {str(e)}")
        profile_path = os.path.join(args.output_dir, "memory_profile_error.json")
        profiler.save_profile(profile_path)
        raise e

    finally:
        # Save final profile
        profile_path = os.path.join(args.output_dir, "memory_profile_complete.json")
        profiler.save_profile(profile_path)
        print(f"Memory profile saved to: {profile_path}")


if __name__ == "__main__":
    main()