"""
Sketch Encoder for Cross-Attention Integration.
Converts sketch edge maps to embeddings compatible with text cross-attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class SketchCrossAttentionEncoder(nn.Module):
    """
    Sketch encoder that generates embeddings for cross-attention integration.
    
    Instead of injecting features at specific U-Net layers, this approach:
    1. Encodes sketch to a sequence of tokens
    2. Combines with text embeddings 
    3. Uses standard cross-attention mechanism
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 512,
        num_sketch_tokens: int = 77,  # Match text token length
        cross_attention_dim: int = 768,  # Match CLIP embedding dim
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_sketch_tokens = num_sketch_tokens
        self.cross_attention_dim = cross_attention_dim
        
        # Convolutional feature extraction
        self.conv_encoder = nn.Sequential(
            # 512x512 -> 256x256
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # 256x256 -> 128x128  
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # 128x128 -> 64x64
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(256, hidden_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Global average pooling to get feature vector
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Project to token sequence
        self.to_tokens = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * num_sketch_tokens),
            nn.ReLU(inplace=True),
        )
        
        # Project to cross-attention dimension
        self.to_cross_attn = nn.Linear(hidden_dim, cross_attention_dim)
        
        # Learnable positional embeddings for sketch tokens
        self.pos_embeddings = nn.Parameter(torch.randn(1, num_sketch_tokens, cross_attention_dim))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, sketch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            sketch: Binary edge map [B, 1, 512, 512]
            
        Returns:
            sketch_embeddings: [B, num_sketch_tokens, cross_attention_dim]
        """
        batch_size = sketch.shape[0]
        
        # Ensure sketch is in [0, 1] range
        sketch = sketch.float()
        if sketch.max() > 1.0:
            sketch = sketch / 255.0
        
        # Extract convolutional features [B, hidden_dim, 32, 32]
        conv_features = self.conv_encoder(sketch)
        
        # Global pooling [B, hidden_dim, 1, 1] -> [B, hidden_dim]
        global_features = self.global_pool(conv_features).flatten(1)
        
        # Generate token sequence [B, hidden_dim * num_tokens]
        token_features = self.to_tokens(global_features)
        
        # Reshape to token sequence [B, num_tokens, hidden_dim]
        token_features = token_features.view(batch_size, self.num_sketch_tokens, self.hidden_dim)
        
        # Project to cross-attention dimension [B, num_tokens, cross_attention_dim]
        sketch_embeddings = self.to_cross_attn(token_features)
        
        # Add positional embeddings
        sketch_embeddings = sketch_embeddings + self.pos_embeddings
        
        return sketch_embeddings


class SketchTextCombiner(nn.Module):
    """
    Combines sketch and text embeddings for joint conditioning.
    """
    
    def __init__(
        self,
        cross_attention_dim: int = 768,
        combination_method: str = "concat",  # "concat", "add", "cross_attn"
    ):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.combination_method = combination_method
        
        if combination_method == "cross_attn":
            # Cross-attention between text and sketch
            self.text_to_sketch_attn = nn.MultiheadAttention(
                embed_dim=cross_attention_dim,
                num_heads=8,
                batch_first=True
            )
            self.sketch_to_text_attn = nn.MultiheadAttention(
                embed_dim=cross_attention_dim,
                num_heads=8,
                batch_first=True
            )
            
            # Layer normalization
            self.text_norm = nn.LayerNorm(cross_attention_dim)
            self.sketch_norm = nn.LayerNorm(cross_attention_dim)
    
    def forward(
        self,
        text_embeddings: torch.Tensor,
        sketch_embeddings: torch.Tensor,
        method: Optional[str] = None
    ) -> torch.Tensor:
        """
        Combine text and sketch embeddings.
        
        Args:
            text_embeddings: [B, text_seq_len, cross_attention_dim]
            sketch_embeddings: [B, sketch_seq_len, cross_attention_dim]
            method: Override default combination method
            
        Returns:
            combined_embeddings: [B, combined_seq_len, cross_attention_dim]
        """
        method = method or self.combination_method
        
        if method == "concat":
            # Simple concatenation along sequence dimension
            return torch.cat([text_embeddings, sketch_embeddings], dim=1)
        
        elif method == "add":
            # Element-wise addition (requires same sequence length)
            if text_embeddings.shape[1] != sketch_embeddings.shape[1]:
                # Interpolate to match lengths
                sketch_embeddings = F.interpolate(
                    sketch_embeddings.transpose(1, 2),
                    size=text_embeddings.shape[1],
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            return text_embeddings + sketch_embeddings
        
        elif method == "cross_attn":
            # Cross-attention between modalities
            text_attended, _ = self.text_to_sketch_attn(
                text_embeddings, sketch_embeddings, sketch_embeddings
            )
            sketch_attended, _ = self.sketch_to_text_attn(
                sketch_embeddings, text_embeddings, text_embeddings
            )
            
            # Residual connections and normalization
            text_out = self.text_norm(text_embeddings + text_attended)
            sketch_out = self.sketch_norm(sketch_embeddings + sketch_attended)
            
            return torch.cat([text_out, sketch_out], dim=1)
        
        else:
            raise ValueError(f"Unknown combination method: {method}")
