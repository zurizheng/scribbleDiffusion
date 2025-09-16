"""
Sketch-conditioned U-Net model for ScribbleDiffusion.
Combines standard diffusion U-Net with ControlNet-lite sketch conditioning.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

# Handle diffusers version compatibility for UNet blocks
try:
    # Try newer diffusers structure (0.21+)
    from diffusers.models.unets.unet_2d_blocks import (
        UNetMidBlock2D,
        get_down_block,
        get_up_block,
        UNetMidBlock2DCrossAttn,
        DownBlock2D,
        CrossAttnDownBlock2D,
        UpBlock2D,
        CrossAttnUpBlock2D
    )
    DIFFUSERS_BLOCKS_AVAILABLE = True
except ImportError:
    try:
        # Try intermediate structure
        from diffusers.models.unet_2d_blocks import (
            UNetMidBlock2D,
            get_down_block,
            get_up_block,
            UNetMidBlock2DCrossAttn,
            DownBlock2D,
            CrossAttnDownBlock2D,
            UpBlock2D,
            CrossAttnUpBlock2D
        )
        DIFFUSERS_BLOCKS_AVAILABLE = True
    except ImportError:
        # Blocks not available - will use base UNet
        DIFFUSERS_BLOCKS_AVAILABLE = False
        print("⚠️ Diffusers blocks not available, falling back to base implementation")

from diffusers.utils import logging

logger = logging.get_logger(__name__)


class SketchConditionedUNet(nn.Module):
    """
    Lightweight U-Net with sketch conditioning via hint features.
    
    Architecture:
    - Input: 32x32x4 latents (from VAE)
    - Conditioning: Text embeddings + hint features from sketch
    - Output: Predicted noise (epsilon prediction)
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        model_channels: int = 192,
        attention_resolutions: List[int] = [2, 4, 8],
        num_res_blocks: int = 2,
        channel_mult: List[int] = [1, 2, 3, 4],
        num_heads: int = 8,
        use_spatial_transformer: bool = True,
        transformer_depth: int = 1,
        context_dim: int = 768,
        use_checkpoint: bool = True,
        **kwargs,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            Timesteps(model_channels, flip_sin_to_cos=True, downscale_freq_shift=1),
            TimestepEmbedding(model_channels, time_embed_dim),
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = []
                out_ch = model_channels * mult
                
                # Check if we need attention at this resolution
                if ds in attention_resolutions and use_spatial_transformer:
                    self.encoder_blocks.append(
                        CrossAttnDownBlock2D(
                            in_channels=ch,
                            out_channels=out_ch,
                            temb_channels=time_embed_dim,
                            num_layers=1,
                            transformer_layers_per_block=transformer_depth,
                            num_attention_heads=num_heads,
                            cross_attention_dim=context_dim,
                            use_linear_projection=False,
                        )
                    )
                else:
                    self.encoder_blocks.append(
                        DownBlock2D(
                            in_channels=ch,
                            out_channels=out_ch,
                            temb_channels=time_embed_dim,
                            num_layers=1,
                        )
                    )
                
                ch = out_ch
                input_block_chans.append(ch)
            
            if level < len(channel_mult) - 1:
                # Downsample
                self.encoder_blocks.append(
                    DownBlock2D(
                        in_channels=ch,
                        out_channels=ch,
                        temb_channels=time_embed_dim,
                        num_layers=1,
                        add_downsample=True,
                    )
                )
                input_block_chans.append(ch)
                ds *= 2
        
        # Middle block
        self.middle_block = UNetMidBlock2DCrossAttn(
            in_channels=ch,
            temb_channels=time_embed_dim,
            transformer_layers_per_block=transformer_depth,
            num_attention_heads=num_heads,
            cross_attention_dim=context_dim,
        )
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                out_ch = model_channels * mult
                
                if ds in attention_resolutions and use_spatial_transformer:
                    self.decoder_blocks.append(
                        CrossAttnUpBlock2D(
                            in_channels=ch + ich,
                            out_channels=out_ch,
                            prev_output_channel=ich,
                            temb_channels=time_embed_dim,
                            num_layers=1,
                            transformer_layers_per_block=transformer_depth,
                            num_attention_heads=num_heads,
                            cross_attention_dim=context_dim,
                            use_linear_projection=False,
                        )
                    )
                else:
                    self.decoder_blocks.append(
                        UpBlock2D(
                            in_channels=ch + ich,
                            out_channels=out_ch,
                            prev_output_channel=ich,
                            temb_channels=time_embed_dim,
                            num_layers=1,
                        )
                    )
                
                ch = out_ch
                
                if level > 0 and i == num_res_blocks:
                    # Upsample
                    self.decoder_blocks[-1].add_upsample = True
                    ds //= 2
        
        # Output convolution
        self.conv_norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)
    
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        hint_features: Optional[Dict[str, torch.Tensor]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        """
        Forward pass of sketch-conditioned U-Net.
        
        Args:
            sample: Input latents [B, 4, H, W]
            timestep: Timestep for diffusion process
            encoder_hidden_states: Text embeddings [B, seq_len, context_dim]
            hint_features: Sketch conditioning features from HintEncoder
            return_dict: Whether to return dict or tensor
            
        Returns:
            Predicted noise or dict with 'sample' key
        """
        # Time embedding
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_embed(timesteps)
        
        # Initial convolution
        h = self.conv_in(sample)
        
        # Add hint features if provided (ControlNet-lite injection)
        if hint_features is not None and "64" in hint_features:
            h = h + hint_features["64"]
        
        # Encoder
        down_block_res_samples = [h]
        for i, encoder_block in enumerate(self.encoder_blocks):
            if hasattr(encoder_block, "has_cross_attention") and encoder_block.has_cross_attention:
                h, res_samples = encoder_block(
                    h, t_emb, encoder_hidden_states=encoder_hidden_states
                )
            else:
                h, res_samples = encoder_block(h, t_emb)
            
            down_block_res_samples.extend(res_samples)
        
        # Middle block
        h = self.middle_block(h, t_emb, encoder_hidden_states=encoder_hidden_states)
        
        # Decoder
        for decoder_block in self.decoder_blocks:
            res_samples = down_block_res_samples[-len(decoder_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(decoder_block.resnets)]
            
            if hasattr(decoder_block, "has_cross_attention") and decoder_block.has_cross_attention:
                h = decoder_block(
                    h, res_samples, t_emb, encoder_hidden_states=encoder_hidden_states
                )
            else:
                h = decoder_block(h, res_samples, t_emb)
        
        # Output
        h = self.conv_norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        if not return_dict:
            return (h,)
        
        return {"sample": h}


class UNetOutput:
    """Output class for U-Net model."""
    
    def __init__(self, sample: torch.FloatTensor):
        self.sample = sample
