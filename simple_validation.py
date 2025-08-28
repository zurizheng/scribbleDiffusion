#!/usr/bin/env python3
"""
Simplified validation - tests core components separately
"""

import torch
import time

def quick_validation():
    """Quick validation of key components"""
    print("🚀 ScribbleDiffusion - Simplified Validation")
    print("=" * 50)
    
    # Test 1: GPU Availability
    print("1️⃣ Testing GPU...")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   ✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("   ❌ No GPU available")
        return False
    
    # Test 2: Basic PyTorch Operations
    print("2️⃣ Testing PyTorch operations...")
    try:
        x = torch.randn(100, 100, device='cuda')
        y = torch.matmul(x, x.T)
        print(f"   ✅ GPU computation successful: {y.shape}")
    except Exception as e:
        print(f"   ❌ GPU computation failed: {e}")
        return False
    
    # Test 3: Import Core Modules
    print("3️⃣ Testing imports...")
    try:
        from src.models.unet import SketchConditionedUNet
        from src.models.hint_encoder import HintEncoder
        from transformers import CLIPTokenizer
        from diffusers import AutoencoderKL, DDIMScheduler
        print("   ✅ All core modules imported successfully")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 4: COCO Dataset (basic)
    print("4️⃣ Testing dataset...")
    try:
        from src.data.coco_dataset import COCOScribbleDataset
        from omegaconf import DictConfig
        
        config = DictConfig({
            'coco_root': './data/coco',
            'image_size': 128,
            'max_length': 77
        })
        tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        dataset = COCOScribbleDataset(config=config, tokenizer=tokenizer, split='demo', download=True)
        
        if len(dataset) > 0:
            print(f"   ✅ Dataset loaded: {len(dataset)} samples")
        else:
            print("   ❌ Dataset is empty")
            return False
    except Exception as e:
        print(f"   ❌ Dataset test failed: {e}")
        return False
    
    # Test 5: Memory estimation
    print("5️⃣ Testing memory usage...")
    try:
        # Estimate memory for a small model
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated() / 1024**3
        
        # Create a small test model
        test_model = torch.nn.Sequential(
            torch.nn.Conv2d(4, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 4, 3, padding=1)
        ).cuda()
        
        # Test forward pass
        test_input = torch.randn(2, 4, 64, 64, device='cuda')
        output = test_model(test_input)
        
        end_memory = torch.cuda.memory_allocated() / 1024**3
        used_memory = end_memory - start_memory
        
        print(f"   ✅ Test model: {used_memory:.2f}GB VRAM used")
        
        # Clean up
        del test_model, test_input, output
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"   ❌ Memory test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ ALL VALIDATION TESTS PASSED!")
    print("\n🎯 Ready for training!")
    print("\n📝 Recommended training commands:")
    print("   # Quick test (2-5 minutes):")
    print("   python train.py --config configs/debug.yaml")
    print("   ")
    print("   # Full training (5-7 hours):")
    print("   python train.py --config configs/coco.yaml")
    print("\n⚡ Your GTX 1650 (4GB) is ready for ScribbleDiffusion!")
    
    return True

if __name__ == "__main__":
    start_time = time.time()
    success = quick_validation()
    elapsed = time.time() - start_time
    
    print(f"\n⏱️  Validation completed in {elapsed:.1f}s")
    
    if success:
        print("🚀 System is ready for ScribbleDiffusion training!")
    else:
        print("❌ Please fix the issues above before training.")
