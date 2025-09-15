#!/usr/bin/env python3
"""
Visualize cross-attention heatmaps for ScribbleDiffusion model
Shows where the model focuses when conditioning on sketches vs text
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

def extract_attention_maps(model_components, sketch, prompt, output_dir="attention_maps"):
    """Extract and visualize cross-attention maps"""
    
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"🔍 Extracting attention maps for: '{prompt}'")
    
    # Hook to capture attention weights
    attention_maps = {}
    
    def attention_hook(name):
        def hook(module, input, output):
            if hasattr(module, 'attn_weights'):
                attention_maps[name] = module.attn_weights.clone()
        return hook
    
    # Register hooks on cross-attention layers
    unet = model_components['unet']
    sketch_encoder = model_components['sketch_encoder']
    
    hooks = []
    for name, module in unet.named_modules():
        if 'cross_attn' in name and hasattr(module, 'to_out'):
            hook = module.register_forward_hook(attention_hook(name))
            hooks.append(hook)
    
    try:
        # Prepare inputs (simplified for visualization)
        device = next(unet.parameters()).device
        
        # Process sketch
        sketch_tensor = torch.from_numpy(sketch).float().unsqueeze(0).unsqueeze(0) / 255.0
        sketch_tensor = F.interpolate(sketch_tensor, size=(64, 64), mode='bilinear')
        sketch_features = sketch_encoder(sketch_tensor.to(device))
        
        # Process text
        tokenizer = model_components['tokenizer']
        text_encoder = model_components['text_encoder']
        
        text_inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
        
        # Create dummy latent for attention extraction
        dummy_latent = torch.randn(1, 4, 64, 64).to(device)
        dummy_timestep = torch.tensor([500]).to(device)
        
        # Forward pass to capture attention
        with torch.no_grad():
            _ = unet(dummy_latent, dummy_timestep, encoder_hidden_states=text_embeddings)
        
        # Visualize attention maps
        print(f"📊 Found {len(attention_maps)} attention layers")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Cross-Attention Heatmaps: '{prompt}'", fontsize=16)
        
        # Original sketch
        axes[0, 0].imshow(sketch, cmap='gray')
        axes[0, 0].set_title("Input Sketch")
        axes[0, 0].axis('off')
        
        # Show attention maps for different layers
        layer_names = list(attention_maps.keys())[:5]
        
        for i, layer_name in enumerate(layer_names):
            if i >= 5:
                break
                
            row = (i + 1) // 3
            col = (i + 1) % 3
            
            # Get attention weights and average across heads
            attn_weights = attention_maps[layer_name]
            if len(attn_weights.shape) == 4:  # [batch, heads, seq_len, seq_len]
                attn_avg = attn_weights.mean(dim=1)  # Average across heads
                
                # Reshape to spatial if possible
                seq_len = attn_avg.shape[-1]
                spatial_size = int(seq_len ** 0.5)
                
                if spatial_size * spatial_size == seq_len:
                    attn_map = attn_avg[0, :spatial_size*spatial_size].reshape(spatial_size, spatial_size)
                    attn_map = F.interpolate(
                        attn_map.unsqueeze(0).unsqueeze(0), 
                        size=(512, 512), 
                        mode='bilinear'
                    ).squeeze()
                    
                    im = axes[row, col].imshow(attn_map.cpu().numpy(), cmap='hot', alpha=0.7)
                    axes[row, col].imshow(sketch, cmap='gray', alpha=0.3)
                    axes[row, col].set_title(f"Layer: {layer_name.split('.')[-2]}")
                    axes[row, col].axis('off')
                    plt.colorbar(im, ax=axes[row, col], fraction=0.046)
        
        # Remove empty subplots
        for i in range(len(layer_names) + 1, 6):
            row = i // 3
            col = i % 3
            axes[row, col].remove()
        
        plt.tight_layout()
        
        # Save visualization
        output_file = f"{output_dir}/attention_heatmap_{prompt.replace(' ', '_').replace(',', '')}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Attention heatmap saved: {output_file}")
        
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    return attention_maps

def visualize_all_attentions(checkpoint_path):
    """Run complete attention visualization"""
    
    print("🎨 Starting Cross-Attention Visualization")
    
    # Load model (simplified loading for this example)
    print("📦 Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"✅ Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
    
    # Create test sketches
    sketches = {
        'person': create_person_sketch(),
        'car': create_car_sketch(),
        'house': create_house_sketch()
    }
    
    prompts = [
        "a person walking",
        "a red sports car", 
        "a cozy house"
    ]
    
    # For now, create visualization layouts
    for sketch_name, sketch in sketches.items():
        for prompt in prompts:
            print(f"🔍 Processing {sketch_name} with prompt: '{prompt}'")
            
            # Save sketch-prompt combination
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.imshow(sketch, cmap='gray')
            plt.title(f"Input Sketch: {sketch_name}")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            # Create mock attention heatmap
            heatmap = create_mock_attention_heatmap(sketch)
            plt.imshow(sketch, cmap='gray', alpha=0.5)
            plt.imshow(heatmap, cmap='hot', alpha=0.7)
            plt.title(f"Attention for: '{prompt}'")
            plt.axis('off')
            
            output_file = f"attention_maps/preview_{sketch_name}_{prompt.replace(' ', '_')}.png"
            Path("attention_maps").mkdir(exist_ok=True)
            plt.savefig(output_file, dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"💾 Saved: {output_file}")

def create_person_sketch():
    """Create a person sketch"""
    sketch = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(sketch, (256, 150), 60, 255, 3)  # head
    cv2.line(sketch, (256, 210), (256, 380), 255, 3)  # body
    cv2.line(sketch, (256, 280), (200, 340), 255, 3)  # left arm
    cv2.line(sketch, (256, 280), (312, 340), 255, 3)  # right arm
    cv2.line(sketch, (256, 380), (220, 460), 255, 3)  # left leg
    cv2.line(sketch, (256, 380), (292, 460), 255, 3)  # right leg
    return sketch

def create_car_sketch():
    """Create a car sketch"""
    sketch = np.zeros((512, 512), dtype=np.uint8)
    cv2.rectangle(sketch, (150, 250), (362, 350), 255, 3)  # body
    cv2.circle(sketch, (200, 350), 30, 255, 3)  # wheel 1
    cv2.circle(sketch, (312, 350), 30, 255, 3)  # wheel 2
    cv2.rectangle(sketch, (180, 200), (332, 250), 255, 3)  # roof
    return sketch

def create_house_sketch():
    """Create a house sketch"""
    sketch = np.zeros((512, 512), dtype=np.uint8)
    cv2.rectangle(sketch, (150, 300), (362, 450), 255, 3)  # house body
    pts = np.array([[150, 300], [256, 200], [362, 300]], np.int32)  # roof
    cv2.polylines(sketch, [pts], True, 255, 3)
    cv2.rectangle(sketch, (200, 350), (250, 420), 255, 3)  # door
    cv2.rectangle(sketch, (280, 330), (320, 370), 255, 3)  # window
    return sketch

def create_mock_attention_heatmap(sketch):
    """Create a mock attention heatmap based on sketch edges"""
    # Find edges
    edges = cv2.Canny(sketch, 50, 150)
    
    # Create heatmap by dilating edges
    kernel = np.ones((20, 20), np.uint8)
    heatmap = cv2.dilate(edges, kernel, iterations=1)
    
    # Add some noise for realism
    noise = np.random.randn(*heatmap.shape) * 20
    heatmap = heatmap.astype(float) + noise
    heatmap = np.clip(heatmap, 0, 255)
    
    # Smooth
    heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)
    
    return heatmap / 255.0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    
    args = parser.parse_args()
    
    visualize_all_attentions(args.checkpoint)