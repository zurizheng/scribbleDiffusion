#!/usr/bin/env python3
"""
Quick validation script for ScribbleDiffusion
Tests all components without long training
"""

import time
import torch
from pathlib import Path

def test_gpu_memory():
    """Test GPU memory and availability"""
    print("🔍 Testing GPU Setup...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - will use CPU (very slow)")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"✅ GPU: {gpu_name}")
    print(f"✅ VRAM: {gpu_memory:.1f}GB")
    
    # Test memory allocation
    try:
        test_tensor = torch.randn(1000, 1000, device='cuda')
        del test_tensor
        torch.cuda.empty_cache()
        print("✅ GPU memory allocation test passed")
        return True
    except Exception as e:
        print(f"❌ GPU memory test failed: {e}")
        return False

def test_model_loading():
    """Test model initialization"""
    print("\n🔍 Testing Model Loading...")
    
    try:
        from src.models.unet import SketchConditionedUNet
        from src.models.hint_encoder import HintEncoder
        
        # Tiny model for testing (use minimal working config)
        unet = SketchConditionedUNet(
            in_channels=4,
            out_channels=4,
            model_channels=96,  # Smaller but compatible
            attention_resolutions=[2, 4],  # Minimal attention
            num_res_blocks=1,
            channel_mult=[1, 2, 3],  # Need at least 3 levels for hint encoder
            num_heads=4,
            use_spatial_transformer=True,
            transformer_depth=1,
            context_dim=768,
            use_checkpoint=False
        )
        
        hint_encoder = HintEncoder(
            in_channels=1,
            hint_channels=[16, 32, 64, 96],  # Match UNet model_channels
            injection_layers=[0, 1, 2, 3],
            injection_method="add"
        )
        
        # Count parameters
        unet_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        hint_params = sum(p.numel() for p in hint_encoder.parameters() if p.requires_grad)
        total_params = unet_params + hint_params
        
        print(f"✅ UNet loaded: {unet_params:,} parameters")
        print(f"✅ HintEncoder loaded: {hint_params:,} parameters") 
        print(f"✅ Total trainable: {total_params:,} parameters")
        
        # Test forward pass
        if torch.cuda.is_available():
            unet = unet.cuda()
            hint_encoder = hint_encoder.cuda()
            
            # Test tensors (corrected sizes)
            latents = torch.randn(1, 4, 32, 32).cuda()  # 32x32 latent space
            sketch = torch.randn(1, 1, 256, 256).cuda()  # 256x256 image space (8x larger)
            timesteps = torch.randint(0, 100, (1,)).cuda()
            text_embeddings = torch.randn(1, 77, 768).cuda()
            
            # Forward pass
            hint_features = hint_encoder(sketch)
            noise_pred = unet(latents, timesteps, text_embeddings, hint_features)
            
            print(f"✅ Forward pass successful: {noise_pred.shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_loading():
    """Test dataset loading"""
    print("\n🔍 Testing Dataset Loading...")
    
    try:
        from src.data.coco_dataset import COCOScribbleDataset
        from transformers import CLIPTokenizer
        from omegaconf import DictConfig
        
        # Simple config
        config = DictConfig({
            'coco_root': './data/coco',
            'image_size': 128,
            'max_length': 77,
            'edge_threshold_low': 50,
            'edge_threshold_high': 150
        })
        
        tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        
        dataset = COCOScribbleDataset(
            config=config,
            tokenizer=tokenizer,
            split='demo',
            download=True
        )
        
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✅ Sample keys: {list(sample.keys())}")
            print(f"✅ Image shape: {sample['images'].shape}")
            print(f"✅ Sketch shape: {sample['sketches'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step():
    """Test a single training step"""
    print("\n🔍 Testing Training Step...")
    
    try:
        from src.models.unet import SketchConditionedUNet
        from src.models.hint_encoder import HintEncoder
        from diffusers import DDIMScheduler
        import torch.nn.functional as F
        
        # Tiny models (use minimal working config)
        unet = SketchConditionedUNet(
            in_channels=4, out_channels=4, model_channels=96,
            attention_resolutions=[2, 4], num_res_blocks=1,
            channel_mult=[1, 2, 3], num_heads=4,
            use_spatial_transformer=True, transformer_depth=1,
            context_dim=768, use_checkpoint=False
        )
        
        hint_encoder = HintEncoder(
            in_channels=1, hint_channels=[16, 32, 64, 96],
            injection_layers=[0, 1, 2, 3], injection_method="add"
        )
        
        scheduler = DDIMScheduler(num_train_timesteps=100)
        
        if torch.cuda.is_available():
            unet = unet.cuda()
            hint_encoder = hint_encoder.cuda()
        
        # Mock batch (corrected sizes)
        batch_size = 2
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        latents = torch.randn(batch_size, 4, 32, 32, device=device)  # 32x32 latent
        sketch = torch.randn(batch_size, 1, 256, 256, device=device)  # 256x256 image
        text_embeddings = torch.randn(batch_size, 77, 768, device=device)
        
        # Add noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 100, (batch_size,), device=device)
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        
        # Forward pass
        hint_features = hint_encoder(sketch)
        noise_pred = unet(noisy_latents, timesteps, text_embeddings, hint_features)
        
        # Loss
        loss = F.mse_loss(noise_pred, noise)
        
        # Backward pass
        loss.backward()
        
        print(f"✅ Training step successful!")
        print(f"✅ Loss: {loss.item():.4f}")
        print(f"✅ Memory used: {torch.cuda.memory_allocated()/1024**3:.2f}GB" if torch.cuda.is_available() else "CPU mode")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests"""
    print("🚀 ScribbleDiffusion Quick Validation")
    print("=" * 50)
    
    start_time = time.time()
    
    tests = [
        test_gpu_memory,
        test_model_loading,
        test_dataset_loading,
        test_training_step,
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    test_names = ["GPU Setup", "Model Loading", "Dataset Loading", "Training Step"]
    for name, result in zip(test_names, results):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:20s}: {status}")
    
    elapsed = time.time() - start_time
    all_passed = all(results)
    
    print(f"\nTime taken: {elapsed:.1f}s")
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Ready for quick debug training:")
        print("   python train.py --config configs/debug.yaml")
        print("\n⚡ Debug training will take ~2-5 minutes")
    else:
        print("\n❌ Some tests failed. Please fix issues before training.")
    
    return all_passed

if __name__ == "__main__":
    main()
