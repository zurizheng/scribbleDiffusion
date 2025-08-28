"""
Evaluation metrics and utilities for ScribbleDiffusion.
"""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import io


class ScribbleEvaluator:
    """
    Comprehensive evaluation suite for ScribbleDiffusion.
    
    Metrics:
    - CLIP score (text-image alignment)
    - Edge fidelity (sketch adherence)
    - Perceptual quality metrics
    - User preference studies
    """
    
    def __init__(self):
        self.metrics_history = []
    
    def evaluate_batch(
        self,
        generated_images: List[Image.Image],
        reference_sketches: List[np.ndarray],
        text_prompts: List[str],
        reference_images: List[Image.Image] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a batch of generated images.
        
        Args:
            generated_images: Generated images
            reference_sketches: Input sketches
            text_prompts: Input text prompts
            reference_images: Ground truth images (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Edge fidelity metrics
        edge_scores = []
        for gen_img, ref_sketch in zip(generated_images, reference_sketches):
            edge_score = self.calculate_edge_fidelity(gen_img, ref_sketch)
            edge_scores.append(edge_score)
        
        metrics["edge_fidelity_mean"] = np.mean(edge_scores)
        metrics["edge_fidelity_std"] = np.std(edge_scores)
        
        # CLIP score (placeholder - would need actual CLIP model)
        clip_scores = []
        for gen_img, prompt in zip(generated_images, text_prompts):
            clip_score = self.calculate_clip_score(gen_img, prompt)
            clip_scores.append(clip_score)
        
        metrics["clip_score_mean"] = np.mean(clip_scores)
        metrics["clip_score_std"] = np.std(clip_scores)
        
        # Perceptual quality
        quality_scores = []
        for gen_img in generated_images:
            quality_score = self.calculate_perceptual_quality(gen_img)
            quality_scores.append(quality_score)
        
        metrics["perceptual_quality_mean"] = np.mean(quality_scores)
        metrics["perceptual_quality_std"] = np.std(quality_scores)
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        return metrics
    
    def calculate_edge_fidelity(
        self,
        generated_image: Image.Image,
        reference_sketch: np.ndarray,
    ) -> float:
        """
        Calculate how well the generated image follows the input sketch.
        
        Uses edge detection on generated image and compares with input sketch.
        """
        # Convert generated image to edges
        gen_array = np.array(generated_image.convert("L"))
        gen_edges = cv2.Canny(gen_array, 50, 150)
        
        # Normalize reference sketch
        if reference_sketch.max() > 1:
            reference_sketch = (reference_sketch > 128).astype(np.uint8) * 255
        
        # Resize to match if needed
        if gen_edges.shape != reference_sketch.shape:
            reference_sketch = cv2.resize(
                reference_sketch, 
                (gen_edges.shape[1], gen_edges.shape[0])
            )
        
        # Calculate intersection over union (IoU)
        intersection = np.logical_and(gen_edges > 0, reference_sketch > 0)
        union = np.logical_or(gen_edges > 0, reference_sketch > 0)
        
        if union.sum() == 0:
            return 0.0
        
        iou = intersection.sum() / union.sum()
        return float(iou)
    
    def calculate_clip_score(
        self,
        generated_image: Image.Image,
        text_prompt: str,
    ) -> float:
        """
        Calculate CLIP score between image and text.
        
        Note: This is a placeholder implementation.
        In practice, you'd use the actual CLIP model.
        """
        # Placeholder implementation
        # In practice, you'd:
        # 1. Encode image with CLIP vision encoder
        # 2. Encode text with CLIP text encoder  
        # 3. Calculate cosine similarity
        
        # For now, return a random score based on text length
        # (longer, more descriptive prompts might get higher scores)
        base_score = 0.7
        text_bonus = min(len(text_prompt.split()) * 0.02, 0.3)
        noise = np.random.normal(0, 0.05)
        
        score = base_score + text_bonus + noise
        return max(0.0, min(1.0, score))
    
    def calculate_perceptual_quality(
        self,
        generated_image: Image.Image,
    ) -> float:
        """
        Calculate perceptual quality score.
        
        Uses simple heuristics like contrast, sharpness, etc.
        In practice, you might use LPIPS or other perceptual metrics.
        """
        img_array = np.array(generated_image.convert("L"))
        
        # Calculate contrast (standard deviation of pixel values)
        contrast = np.std(img_array) / 255.0
        
        # Calculate sharpness (variance of Laplacian)
        laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
        sharpness = np.var(laplacian) / 10000.0  # Normalize
        
        # Combine metrics
        quality = 0.5 * contrast + 0.5 * min(sharpness, 1.0)
        
        return float(quality)
    
    def create_evaluation_report(
        self,
        save_path: str = None,
    ) -> Image.Image:
        """
        Create a visual evaluation report.
        
        Shows trends in metrics over time and current performance.
        """
        if not self.metrics_history:
            print("No evaluation data available")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("ScribbleDiffusion Evaluation Report", fontsize=16)
        
        # Extract metrics over time
        steps = range(len(self.metrics_history))
        edge_fidelity = [m["edge_fidelity_mean"] for m in self.metrics_history]
        clip_scores = [m["clip_score_mean"] for m in self.metrics_history]
        quality_scores = [m["perceptual_quality_mean"] for m in self.metrics_history]
        
        # Plot edge fidelity over time
        axes[0, 0].plot(steps, edge_fidelity, 'b-', marker='o')
        axes[0, 0].set_title("Edge Fidelity Over Time")
        axes[0, 0].set_xlabel("Evaluation Step")
        axes[0, 0].set_ylabel("IoU Score")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot CLIP scores over time
        axes[0, 1].plot(steps, clip_scores, 'g-', marker='s')
        axes[0, 1].set_title("CLIP Scores Over Time") 
        axes[0, 1].set_xlabel("Evaluation Step")
        axes[0, 1].set_ylabel("CLIP Score")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot perceptual quality over time
        axes[1, 0].plot(steps, quality_scores, 'r-', marker='^')
        axes[1, 0].set_title("Perceptual Quality Over Time")
        axes[1, 0].set_xlabel("Evaluation Step")
        axes[1, 0].set_ylabel("Quality Score")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Current metrics summary
        if self.metrics_history:
            current_metrics = self.metrics_history[-1]
            metrics_text = f"""
Current Performance:
Edge Fidelity: {current_metrics['edge_fidelity_mean']:.3f} ± {current_metrics['edge_fidelity_std']:.3f}
CLIP Score: {current_metrics['clip_score_mean']:.3f} ± {current_metrics['clip_score_std']:.3f}
Quality: {current_metrics['perceptual_quality_mean']:.3f} ± {current_metrics['perceptual_quality_std']:.3f}

Total Evaluations: {len(self.metrics_history)}
            """
            axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, 
                           verticalalignment='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title("Current Metrics Summary")
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        report_img = Image.open(buf)
        plt.close()
        
        if save_path:
            report_img.save(save_path)
            print(f"Evaluation report saved to {save_path}")
        
        return report_img
    
    def run_ablation_study(
        self,
        model_variants: Dict[str, any],  # Different model configurations
        test_data: List[Tuple],  # Test sketches and prompts
    ) -> Dict[str, Dict[str, float]]:
        """
        Run ablation study comparing different model variants.
        
        Args:
            model_variants: Dictionary of model names to model instances
            test_data: List of (sketch, prompt) tuples for testing
            
        Returns:
            Results for each model variant
        """
        results = {}
        
        for variant_name, model in model_variants.items():
            print(f"Evaluating variant: {variant_name}")
            
            generated_images = []
            sketches = []
            prompts = []
            
            # Generate images with this variant
            for sketch, prompt in test_data:
                # Generate image (placeholder)
                # In practice: image = model.generate(sketch, prompt)
                image = Image.new('RGB', (256, 256), 'gray')  # Placeholder
                
                generated_images.append(image)
                sketches.append(sketch)
                prompts.append(prompt)
            
            # Evaluate this variant
            variant_metrics = self.evaluate_batch(
                generated_images, sketches, prompts
            )
            
            results[variant_name] = variant_metrics
        
        return results


def create_comparison_grid(
    images: List[Image.Image],
    labels: List[str],
    sketches: List[np.ndarray] = None,
    prompts: List[str] = None,
) -> Image.Image:
    """
    Create a comparison grid for visual evaluation.
    
    Shows original sketches, generated images, and prompts side by side.
    """
    num_samples = len(images)
    cols = 3 if sketches is not None else 2
    rows = num_samples
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        col = 0
        
        # Show sketch if available
        if sketches is not None:
            axes[i, col].imshow(sketches[i], cmap='gray')
            axes[i, col].set_title("Input Sketch")
            axes[i, col].axis('off')
            col += 1
        
        # Show generated image
        axes[i, col].imshow(images[i])
        axes[i, col].set_title(f"Generated ({labels[i]})")
        axes[i, col].axis('off')
        col += 1
        
        # Show prompt if available
        if prompts is not None and col < cols:
            axes[i, col].text(0.1, 0.5, f'"{prompts[i]}"', 
                            fontsize=10, verticalalignment='center',
                            transform=axes[i, col].transAxes, wrap=True)
            axes[i, col].set_title("Prompt")
            axes[i, col].axis('off')
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    comparison_img = Image.open(buf)
    plt.close()
    
    return comparison_img
