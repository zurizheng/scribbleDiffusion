"""
Attention visualization utilities for ScribbleDiffusion.
Creates heatmaps showing which words attend to which image regions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from typing import Dict, List, Tuple, Optional
import io


class AttentionVisualizer:
    """
    Visualizes cross-attention maps between text tokens and image regions.
    
    Features:
    - Word-to-region heatmaps
    - Token importance over time
    - Interactive attention exploration
    """
    
    def __init__(self):
        self.colormap = 'hot'
        self.alpha = 0.6
    
    def extract_attention_maps(
        self,
        attention_outputs: Dict[str, torch.Tensor],
        text_tokens: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Extract and process attention maps from model outputs.
        
        Args:
            attention_outputs: Raw attention tensors from model
            text_tokens: List of text tokens
            
        Returns:
            Processed attention maps for each token
        """
        processed_maps = {}
        
        for step_key, attention_tensor in attention_outputs.items():
            # attention_tensor shape: [batch, heads, seq_len, spatial_tokens]
            # Average across heads and batch
            if len(attention_tensor.shape) == 4:
                attention_map = attention_tensor.mean(dim=(0, 1))  # [seq_len, spatial_tokens]
            else:
                attention_map = attention_tensor.mean(dim=0)  # Handle different shapes
            
            # Convert spatial tokens back to 2D grid
            spatial_size = int(np.sqrt(attention_map.shape[-1]))
            attention_2d = attention_map.view(-1, spatial_size, spatial_size)
            
            # Store for each token
            token_maps = {}
            for i, token in enumerate(text_tokens):
                if i < attention_2d.shape[0]:
                    token_maps[token] = attention_2d[i].cpu().numpy()
            
            processed_maps[step_key] = token_maps
        
        return processed_maps
    
    def create_attention_grid(
        self,
        attention_maps: Dict[str, torch.Tensor],
        prompt: str,
        generated_image: Image.Image,
        max_tokens: int = 6,
    ) -> Image.Image:
        """
        Create a grid showing attention for top tokens.
        
        Args:
            attention_maps: Attention maps from model
            prompt: Original text prompt
            generated_image: Generated image
            max_tokens: Maximum number of tokens to visualize
            
        Returns:
            Grid image showing attention heatmaps
        """
        # Parse tokens from prompt
        tokens = prompt.split()[:max_tokens]
        
        # If we have actual attention maps, process them
        if attention_maps and len(attention_maps) > 0:
            # Use the last step's attention maps
            last_step = list(attention_maps.keys())[-1]
            step_maps = attention_maps[last_step]
            
            # Extract maps for our tokens
            token_attention_maps = self.extract_attention_maps(
                {last_step: step_maps}, tokens
            )[last_step]
        else:
            # Create dummy attention maps for demo
            token_attention_maps = self.create_dummy_attention_maps(tokens)
        
        # Create visualization grid
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Attention Maps for: "{prompt}"', fontsize=16)
        
        # Resize generated image for overlay
        img_array = np.array(generated_image.resize((64, 64)))
        
        for i, (ax, token) in enumerate(zip(axes.flat, tokens)):
            if token in token_attention_maps:
                attention_map = token_attention_maps[token]
                
                # Normalize attention map
                attention_map = (attention_map - attention_map.min()) / (
                    attention_map.max() - attention_map.min() + 1e-8
                )
                
                # Resize to match image
                attention_resized = F.interpolate(
                    torch.from_numpy(attention_map).unsqueeze(0).unsqueeze(0),
                    size=(64, 64),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().numpy()
                
                # Show image with attention overlay
                ax.imshow(img_array)
                ax.imshow(attention_resized, cmap=self.colormap, alpha=self.alpha)
                ax.set_title(f'"{token}"', fontsize=14, fontweight='bold')
                ax.axis('off')
            else:
                ax.axis('off')
        
        # Hide empty subplots
        for i in range(len(tokens), len(axes.flat)):
            axes.flat[i].axis('off')
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        attention_img = Image.open(buf)
        plt.close()
        
        return attention_img
    
    def create_dummy_attention_maps(self, tokens: List[str]) -> Dict[str, np.ndarray]:
        """Create dummy attention maps for demo purposes."""
        attention_maps = {}
        
        for i, token in enumerate(tokens):
            # Create different patterns for different tokens
            if i % 3 == 0:
                # Center attention
                attention_map = self.create_center_attention(16, 16)
            elif i % 3 == 1:
                # Corner attention
                attention_map = self.create_corner_attention(16, 16)
            else:
                # Random attention
                attention_map = np.random.random((16, 16))
            
            attention_maps[token] = attention_map
        
        return attention_maps
    
    def create_center_attention(self, h: int, w: int) -> np.ndarray:
        """Create attention focused on center."""
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        attention = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(h, w) / 4)**2))
        return attention
    
    def create_corner_attention(self, h: int, w: int) -> np.ndarray:
        """Create attention focused on corners."""
        attention = np.zeros((h, w))
        corner_size = min(h, w) // 4
        
        # Top-left corner
        attention[:corner_size, :corner_size] = 1.0
        
        # Add some noise
        attention += np.random.random((h, w)) * 0.3
        
        return attention
    
    def create_token_importance_timeline(
        self,
        attention_maps: Dict[str, Dict[str, np.ndarray]],
        tokens: List[str],
    ) -> Image.Image:
        """
        Create timeline showing token importance across denoising steps.
        
        Args:
            attention_maps: Attention maps for each step and token
            tokens: List of tokens to track
            
        Returns:
            Timeline visualization
        """
        # Calculate token importance (sum of attention) for each step
        steps = sorted(attention_maps.keys())
        token_importance = {token: [] for token in tokens}
        
        for step in steps:
            step_maps = attention_maps[step]
            for token in tokens:
                if token in step_maps:
                    importance = step_maps[token].sum()
                    token_importance[token].append(importance)
                else:
                    token_importance[token].append(0)
        
        # Create timeline plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for token in tokens:
            ax.plot(range(len(steps)), token_importance[token], 
                   marker='o', label=f'"{token}"', linewidth=2)
        
        ax.set_xlabel('Denoising Step')
        ax.set_ylabel('Token Importance')
        ax.set_title('Token Attention Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        timeline_img = Image.open(buf)
        plt.close()
        
        return timeline_img
    
    def create_interactive_attention_map(
        self,
        attention_map: np.ndarray,
        generated_image: Image.Image,
        token: str,
    ) -> Image.Image:
        """
        Create an interactive attention map for a single token.
        
        Args:
            attention_map: 2D attention map
            generated_image: Generated image
            token: Token name
            
        Returns:
            Attention visualization
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        ax1.imshow(generated_image)
        ax1.set_title('Generated Image')
        ax1.axis('off')
        
        # Attention heatmap alone
        im2 = ax2.imshow(attention_map, cmap=self.colormap)
        ax2.set_title(f'Attention for "{token}"')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # Overlay
        img_array = np.array(generated_image.resize(attention_map.shape[::-1]))
        ax3.imshow(img_array)
        ax3.imshow(attention_map, cmap=self.colormap, alpha=self.alpha)
        ax3.set_title('Overlay')
        ax3.axis('off')
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        result_img = Image.open(buf)
        plt.close()
        
        return result_img
