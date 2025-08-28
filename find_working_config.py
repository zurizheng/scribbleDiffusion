#!/usr/bin/env python3
"""
Test different UNet configurations to find one that works on 4GB VRAM
and determine the exact channel progression for HintEncoder matching.
"""

import torch
import yaml
from src.models.unet import SketchConditionedUNet

def test_config(model_channels, channel_mult, name):
    print(f"\n=== Testing {name} ===")
    print(f"model_channels={model_channels}, channel_mult={channel_mult}")
    
    config = {
        'in_channels': 4,
        'out_channels': 4,
        'model_channels': model_channels,
        'attention_resolutions': [2, 4, 8],
        'num_res_blocks': 1,  # Minimal for memory
        'channel_mult': channel_mult,
        'num_heads': 4,
        'use_spatial_transformer': True,
        'transformer_depth': 1,
        'context_dim': 768,
        'use_checkpoint': True
    }
    
    try:
        unet = SketchConditionedUNet(**config)
        
        # Test forward pass
        x = torch.randn(1, 4, 32, 32)  # Small batch for memory
        t = torch.randint(0, 1000, (1,))
        encoder_hidden_states = torch.randn(1, 77, 768)
        
        with torch.no_grad():
            # Trace channel progression manually
            h = unet.conv_in(x)
            print(f"After conv_in: {h.shape}")
            
            expected_channels = []
            for i, block in enumerate(unet.encoder_blocks):
                res_before = h.shape[-1] 
                ch_before = h.shape[1]
                print(f"Block {i}: res={res_before}, channels={ch_before}")
                expected_channels.append(ch_before)
                
                # Simulate block processing (without full forward)
                if hasattr(block, 'downsamplers') and block.downsamplers:
                    # This block downsamples
                    new_res = max(1, res_before // 2)
                    if hasattr(block, 'resnets') and len(block.resnets) > 0:
                        new_ch = block.resnets[-1].out_channels
                    else:
                        new_ch = ch_before
                    h = torch.randn(1, new_ch, new_res, new_res)
                else:
                    # No downsampling, just channel change
                    if hasattr(block, 'resnets') and len(block.resnets) > 0:
                        new_ch = block.resnets[-1].out_channels
                        h = torch.randn(1, new_ch, res_before, res_before)
        
        print(f"SUCCESS! Expected hint channels for resolutions [32,16,8,4]: {expected_channels[:4]}")
        return expected_channels[:4]
        
    except Exception as e:
        print(f"FAILED: {e}")
        return None

# Test progressively smaller configurations
configs_to_test = [
    (160, [1, 2, 4, 4], "Tiny 160"),
    (128, [1, 2, 4, 4], "Micro 128"),  
    (96, [1, 2, 4, 4], "Nano 96"),
    (64, [1, 2, 4, 4], "Pico 64"),
]

working_config = None
for model_ch, ch_mult, name in configs_to_test:
    expected_channels = test_config(model_ch, ch_mult, name)
    if expected_channels is not None:
        working_config = (model_ch, ch_mult, expected_channels)
        break

if working_config:
    model_ch, ch_mult, expected_ch = working_config
    print(f"\n🎉 FOUND WORKING CONFIG:")
    print(f"   model_channels: {model_ch}")
    print(f"   channel_mult: {ch_mult}")
    print(f"   Expected hint channels: {expected_ch}")
    print(f"\nUpdate config:")
    print(f"   model_channels: {model_ch}")
    print(f"   hint_channels: {[max(4, ch//8) for ch in expected_ch]}  # Scaled down")
else:
    print("❌ No working configuration found")
