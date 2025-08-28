"""
Hint Encoder for sketch conditioning (ControlNet-lite).
Converts binary edge maps to multi-resolution feature maps.        
        Args:
            sketch: Binary edge map [B, 1, 512, 512] (updated for high-resolution)
            
        Returns:
            Dict mapping resolution strings to feature tensors
            e.g., {"64": tensor[B, 192, 64, 64], "32": tensor[B, 384, 32, 32], ...}
        """
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HintEncoder(nn.Module):
    """
    Lightweight hint encoder that converts sketch edge maps to 
    multi-resolution conditioning features.
    
    This is our "ControlNet-lite" - much smaller than full ControlNet
    but still provides effective sketch conditioning.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        hint_channels: List[int] = [16, 32, 64, 128],
        injection_layers: List[int] = [0, 1, 2, 3],
        injection_method: str = "add",
        unet_channels: Optional[List[int]] = None,  # NEW: UNet channels to match
    ):
        """
        Args:
            in_channels: Input channels (1 for binary edge map)
            hint_channels: Output channels at each resolution
            injection_layers: Which U-Net layers to inject into
            injection_method: "add" or "film" (Feature-wise Linear Modulation)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hint_channels = hint_channels
        self.injection_layers = injection_layers
        self.injection_method = injection_method
        
        # Build encoder blocks for different resolutions
        # We'll create conv blocks that downsample the sketch
        # to match U-Net feature map resolutions: 64, 32, 16, 8 (for 512x512 input)
        
        self.blocks = nn.ModuleList()
        
        # Block 0: 512 -> 64 (8x downsample)
        self.blocks.append(
            nn.Sequential(
                nn.Conv2d(in_channels, hint_channels[0], 7, stride=8, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(hint_channels[0], hint_channels[0], 3, padding=1),
                nn.ReLU(inplace=True),
            )
        )
        
        # Block 1: 64 -> 32 (2x downsample)
        self.blocks.append(
            nn.Sequential(
                nn.Conv2d(hint_channels[0], hint_channels[1], 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hint_channels[1], hint_channels[1], 3, padding=1),
                nn.ReLU(inplace=True),
            )
        )
        
        # Block 2: 32 -> 16 (2x downsample)
        self.blocks.append(
            nn.Sequential(
                nn.Conv2d(hint_channels[1], hint_channels[2], 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hint_channels[2], hint_channels[2], 3, padding=1),
                nn.ReLU(inplace=True),
            )
        )
        
        # Block 3: 16 -> 8 (2x downsample)
        self.blocks.append(
            nn.Sequential(
                nn.Conv2d(hint_channels[2], hint_channels[3], 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hint_channels[3], hint_channels[3], 3, padding=1),
                nn.ReLU(inplace=True),
            )
        )
        
        # Zero convolutions for ControlNet-style initialization
        # These ensure the hint encoder starts with zero influence
        self.zero_convs = nn.ModuleList()
        
        # Calculate expected U-Net channels at each resolution
        if unet_channels is None:
            # Default to standard configuration - first few blocks use model_channels
            unet_channels = [192, 192, 192, 384]  # Match actual UNet progression
        
        for i, channels in enumerate(hint_channels):
            if i in injection_layers:
                target_channels = unet_channels[i] if i < len(unet_channels) else channels * 2
                self.zero_convs.append(
                    nn.Conv2d(channels, target_channels, 1, padding=0)
                )
            else:
                self.zero_convs.append(None)
        
        # Initialize zero convs with zeros (ControlNet style)
        for zero_conv in self.zero_convs:
            if zero_conv is not None:
                nn.init.zeros_(zero_conv.weight)
                nn.init.zeros_(zero_conv.bias)
    
    def forward(self, sketch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of hint encoder.
        
        Args:
            sketch: Binary edge map [B, 1, 256, 256]
            
        Returns:
            Dict mapping resolution strings to feature tensors
            e.g., {"32": tensor[B, 192, 32, 32], "16": tensor[B, 384, 16, 16], ...}
        """
        # Ensure sketch is float and in range [0, 1]
        sketch = sketch.float()
        if sketch.max() > 1.0:
            sketch = sketch / 255.0
        
        features = {}
        x = sketch
        resolutions = [64, 32, 16, 8]  # Updated for 512x512 input -> 64x64 latents
        
        for i, (block, zero_conv, res) in enumerate(zip(self.blocks, self.zero_convs, resolutions)):
            x = block(x)
            
            if i in self.injection_layers and zero_conv is not None:
                # Apply zero convolution and store
                hint_feature = zero_conv(x)
                features[str(res)] = hint_feature
        
        return features


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer for sketch conditioning."""
    
    def __init__(self, hint_channels: int, feature_channels: int):
        super().__init__()
        self.scale_conv = nn.Conv2d(hint_channels, feature_channels, 1)
        self.shift_conv = nn.Conv2d(hint_channels, feature_channels, 1)
        
        # Initialize to apply no modulation initially
        nn.init.zeros_(self.scale_conv.weight)
        nn.init.ones_(self.scale_conv.bias)
        nn.init.zeros_(self.shift_conv.weight)
        nn.init.zeros_(self.shift_conv.bias)
    
    def forward(self, features: torch.Tensor, hint: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation: features * (1 + scale) + shift"""
        scale = self.scale_conv(hint)
        shift = self.shift_conv(hint)
        return features * (1 + scale) + shift
