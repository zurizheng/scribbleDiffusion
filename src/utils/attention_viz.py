#!/usr/bin/env python3
"""
Enhanced attention visualization integrated into the web interface
with better organization and intuitive terminology
"""

import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import shutil
from datetime import datetime
import sys
import hashlib

# Add the source directory to path
sys.path.append('src')
sys.path.append('scripts')

from fixed_inference import FixedScribblePipeline

class AttentionVisualizer:
    """Enhanced attention visualization with gallery organization"""
    
    def __init__(self, gallery_root="gallery"):
        self.gallery_root = gallery_root
        self.pipeline = None
        
    def create_prompt_folder(self, prompt, sketch_info=""):
        """Create organized folder for prompt results"""
        # Create safe folder name
        safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_prompt = safe_prompt.replace(' ', '_')[:50]  # Limit length
        
        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{safe_prompt}_{timestamp}"
        
        folder_path = os.path.join(self.gallery_root, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        # Save prompt info
        with open(os.path.join(folder_path, "prompt_info.txt"), "w") as f:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if sketch_info:
                f.write(f"Sketch: {sketch_info}\n")
        
        return folder_path
    
    def get_sketch_feature_description(self, token_idx, attention_map, sketch_image):
        """Get description of what sketch feature a token represents"""
        if token_idx < 77:
            return f"Text feature {token_idx}"
        
        sketch_token_idx = token_idx - 77
        
        # Analyze attention pattern to describe sketch feature
        attention_2d = attention_map.reshape(32, 32)
        
        # Find peak attention location
        peak_y, peak_x = np.unravel_index(np.argmax(attention_2d), attention_2d.shape)
        
        # Convert to image coordinates (32x32 -> 256x256)
        img_y = int(peak_y * 8)  # 256/32 = 8
        img_x = int(peak_x * 8)
        
        # Analyze sketch content around peak
        sketch_array = np.array(sketch_image.convert('L'))
        
        # Check 16x16 region around peak
        y_start = max(0, img_y - 8)
        y_end = min(256, img_y + 8)
        x_start = max(0, img_x - 8)
        x_end = min(256, img_x + 8)
        
        region = sketch_array[y_start:y_end, x_start:x_end]
        avg_intensity = region.mean()
        
        # Describe the feature based on location and intensity
        location_desc = ""
        if peak_y < 10:
            location_desc += "top"
        elif peak_y > 22:
            location_desc += "bottom"
        else:
            location_desc += "center"
            
        if peak_x < 10:
            location_desc += "-left"
        elif peak_x > 22:
            location_desc += "-right"
        else:
            location_desc += "-middle"
        
        if avg_intensity > 200:
            intensity_desc = "bright lines"
        elif avg_intensity > 100:
            intensity_desc = "edges"
        else:
            intensity_desc = "background"
        
        return f"Sketch feature {sketch_token_idx}: {location_desc} {intensity_desc}"
    
    def create_comprehensive_attention_analysis(self, prompt, sketch_image, result, folder_path):
        """Create comprehensive attention analysis with all visualizations"""
        
        attention_maps = result.attention_maps
        if not attention_maps:
            print("⚠️ No attention maps available")
            return {}
        
        # Sort noise levels (high to low)
        noise_levels = sorted([int(k.split('_')[-1]) for k in attention_maps.keys()], reverse=True)
        max_noise = noise_levels[0] if noise_levels else 1000
        
        print(f"📊 Creating comprehensive analysis for {len(noise_levels)} denoising stages...")
        
        # Prepare evolution data
        evolution_data = []
        evolution_stats = []
        
        for i, noise_level in enumerate(noise_levels):
            step_key = f'cross_attention_step_{noise_level}'
            attention = attention_maps[step_key]  # (8, 1024, 154)
            
            # Average across heads
            attention_avg = attention.mean(axis=0)  # (1024, 154)
            
            # Find most active token
            token_activity = attention_avg.max(axis=0)
            most_active_token = np.argmax(token_activity)
            token_attention = attention_avg[:, most_active_token]
            attention_2d = token_attention.reshape(32, 32)
            
            # Calculate denoising progress (0% = max noise, 100% = clean)
            denoising_progress = i / (len(noise_levels) - 1) if len(noise_levels) > 1 else 1.0
            
            # Get feature description
            feature_desc = self.get_sketch_feature_description(
                most_active_token, token_attention, sketch_image
            )
            
            evolution_data.append(attention_2d)
            evolution_stats.append({
                'noise_level': noise_level,
                'progress': denoising_progress,
                'token': most_active_token,
                'feature_desc': feature_desc,
                'mean': token_attention.mean(),
                'max': token_attention.max(),
                'std': token_attention.std()
            })
            
            print(f"   Stage {i+1}: {denoising_progress:.0%} clean → {feature_desc}")
        
        # 1. ENHANCED EVOLUTION GRID
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, (attention_map, stats) in enumerate(zip(evolution_data, evolution_stats)):
            if i < len(axes):
                ax = axes[i]
                
                # Apply gamma correction for better contrast
                gamma = 0.5
                enhanced_map = np.power(attention_map / attention_map.max(), gamma)
                
                im = ax.imshow(enhanced_map, cmap='hot', interpolation='bilinear')
                
                progress_pct = stats['progress'] * 100
                # Clean up the title - remove \\n issues
                title_lines = [
                    f'Progress: {progress_pct:.0f}% Clean',
                    f'Noise Remaining: {stats["noise_level"]}',
                    stats['feature_desc'].split(':')[1].strip() if ':' in stats['feature_desc'] else stats['feature_desc'],
                    f'Peak: {stats["max"]:.4f}'
                ]
                ax.set_title('\n'.join(title_lines), fontsize=9)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide extra subplots
        for i in range(len(evolution_data), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Real Attention Evolution: "{prompt}"\nHigh Noise → Clean Image', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        grid_path = os.path.join(folder_path, 'attention_evolution_grid.png')
        plt.savefig(grid_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. ENHANCED ANIMATED GIF
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            stats = evolution_stats[frame]
            attention_map = evolution_data[frame]
            
            # Left: Enhanced attention map
            gamma = 0.5
            enhanced_map = np.power(attention_map / attention_map.max(), gamma)
            im1 = ax1.imshow(enhanced_map, cmap='hot', interpolation='bilinear')
            
            progress_pct = stats['progress'] * 100
            
            # Clean title without \n display issues
            title_text = f'Denoising Progress: {progress_pct:.0f}% Clean\n'
            title_text += f'Noise Remaining: {stats["noise_level"]}\n'
            title_text += stats['feature_desc']
            
            ax1.set_title(title_text, fontsize=11, fontweight='bold')
            ax1.axis('off')
            
            # Right: Statistics plot
            frame_indices = list(range(len(evolution_stats)))
            means = [s['mean'] for s in evolution_stats]
            maxes = [s['max'] for s in evolution_stats]
            progress_values = [s['progress'] * 100 for s in evolution_stats]
            
            ax2.plot(progress_values, means, 'b-o', label='Mean Attention', linewidth=2, markersize=4)
            ax2.plot(progress_values, maxes, 'r-s', label='Peak Attention', linewidth=2, markersize=4)
            
            # Current position
            current_progress = stats['progress'] * 100
            ax2.axvline(current_progress, color='green', linestyle='--', alpha=0.7, linewidth=2)
            
            ax2.set_xlabel('Denoising Progress (%)')
            ax2.set_ylabel('Attention Value')
            ax2.set_title('Attention Strength During Denoising')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, 100)
            
            return [im1]
        
        anim = animation.FuncAnimation(fig, animate, frames=len(evolution_data), 
                                      interval=1200, blit=False, repeat=True)
        
        gif_path = os.path.join(folder_path, 'attention_evolution.gif')
        anim.save(gif_path, writer='pillow', fps=0.8)
        plt.close()
        
        # 3. TEXT vs SKETCH ATTENTION COMPARISON
        final_attention = attention_maps[f'cross_attention_step_{noise_levels[-1]}'].mean(axis=0)
        
        # Find top text and sketch tokens
        text_activity = final_attention[:, :77].max(axis=0)
        sketch_activity = final_attention[:, 77:].max(axis=0)
        
        top_text_indices = np.argsort(text_activity)[::-1][:3]
        top_sketch_indices = np.argsort(sketch_activity)[::-1][:3] + 77
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Text tokens
        for i in range(3):
            ax = axes[0, i]
            if i < len(top_text_indices):
                token_idx = top_text_indices[i]
                token_attention = final_attention[:, token_idx].reshape(32, 32)
                
                im = ax.imshow(token_attention, cmap='hot', interpolation='nearest')
                ax.set_title(f'Text Token {token_idx}\nActivity: {text_activity[token_idx]:.4f}', fontsize=10)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis('off')
        
        # Sketch tokens
        for i in range(3):
            ax = axes[1, i]
            if i < len(top_sketch_indices):
                token_idx = top_sketch_indices[i]
                token_attention = final_attention[:, token_idx].reshape(32, 32)
                feature_desc = self.get_sketch_feature_description(token_idx, token_attention.flatten(), sketch_image)
                
                im = ax.imshow(token_attention, cmap='plasma', interpolation='nearest')
                ax.set_title(f'{feature_desc}\nActivity: {sketch_activity[token_idx-77]:.4f}', fontsize=10)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis('off')
        
        plt.suptitle(f'Text vs Sketch Attention: "{prompt}"', fontsize=14, fontweight='bold')
        plt.tight_layout()
        comparison_path = os.path.join(folder_path, 'text_vs_sketch_attention.png')
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save summary statistics
        stats_path = os.path.join(folder_path, 'attention_stats.txt')
        with open(stats_path, 'w') as f:
            f.write(f"Attention Analysis for: {prompt}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            total_text_att = final_attention[:, :77].sum()
            total_sketch_att = final_attention[:, 77:].sum()
            total_att = total_text_att + total_sketch_att
            
            f.write(f"Overall Attention Balance:\n")
            f.write(f"Text attention: {(total_text_att/total_att)*100:.1f}%\n")
            f.write(f"Sketch attention: {(total_sketch_att/total_att)*100:.1f}%\n")
            f.write(f"Sketch dominance: {(total_sketch_att/total_text_att):.1f}x stronger\n\n")
            
            f.write(f"Evolution Summary ({len(evolution_stats)} stages):\n")
            for i, stats in enumerate(evolution_stats):
                f.write(f"Stage {i+1}: {stats['progress']*100:.0f}% clean - {stats['feature_desc']}\n")
        
        print(f"✅ Saved comprehensive analysis to: {folder_path}")
        
        return {
            'folder_path': folder_path,
            'evolution_grid': grid_path,
            'evolution_gif': gif_path,
            'text_vs_sketch': comparison_path,
            'stats_file': stats_path,
            'evolution_stats': evolution_stats
        }

def test_enhanced_visualization():
    """Test the enhanced visualization system"""
    print("🧪 Testing enhanced attention visualization system...")
    
    # Initialize
    viz = EnhancedAttentionVisualizer()
    pipeline = FixedScribblePipeline(force_cpu=False)
    
    # Create test sketch
    test_sketch = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(test_sketch, (128, 128), 40, 255, 3)
    cv2.rectangle(test_sketch, (90, 90), (166, 166), 255, 2)
    test_sketch_pil = Image.fromarray(test_sketch)
    
    prompt = "red apple with green leaf"
    
    # Generate with attention
    result = pipeline.generate(
        prompt=prompt,
        sketch_path=test_sketch_pil,
        num_inference_steps=6,
        seed=42,
        return_attention_maps=True
    )
    
    # Create comprehensive analysis
    analysis = viz.create_comprehensive_attention_analysis(
        prompt, test_sketch_pil, result, viz.create_prompt_folder(prompt)
    )
    
        print(f"\n🎉 Enhanced visualization test complete!")
        print(f"Generated files:")
        for key, path in analysis.items():
            if isinstance(path, str) and os.path.exists(path):
                print(f"   {key}: {os.path.basename(path)}")    return analysis

if __name__ == "__main__":
    test_enhanced_visualization()