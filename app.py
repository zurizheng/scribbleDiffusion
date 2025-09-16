"""
Demo web application for ScribbleDiffusion.
Interactive interface for sketch-to-image generation with attention visualization.
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2
from pathlib import Path
import io
import base64
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

# Import our models (you'll need to adjust paths as needed)
from src.models.unet import SketchConditionedUNet
from src.models.hint_encoder import HintEncoder
from src.inference.pipeline import ScribbleDiffusionPipeline
from src.utils.device_utils import get_optimal_device
from src.utils.attention_viz import AttentionVisualizer


class ScribbleDiffusionDemo:
    """
    Interactive demo application for ScribbleDiffusion.
    
    Features:
    - Canvas for sketch input
    - Text prompt input
    - Dual guidance controls (text/sketch)
    - Live attention visualization
    - Denoising timeline
    """
    
    def __init__(self, model_path: str):
        """Initialize demo with trained model."""
        self.device = get_optimal_device()
        self.pipeline = self.load_pipeline(model_path)
        self.attention_viz = AttentionVisualizer()
        
    def load_pipeline(self, model_path: str) -> ScribbleDiffusionPipeline:
        """Load the trained ScribbleDiffusion pipeline."""
        # This is a placeholder - you'll implement the actual loading
        # based on your checkpoint format
        
        # For now, return a dummy pipeline
        print(f"Loading model from {model_path}")
        print("Note: Model loading not fully implemented yet")
        
        # You would load your trained weights here
        # pipeline = ScribbleDiffusionPipeline.from_pretrained(model_path)
        # return pipeline
        
        return None  # Placeholder
    
    def process_sketch(self, sketch_input) -> np.ndarray:
        """Process sketch input from canvas."""
        if sketch_input is None:
            return np.zeros((256, 256), dtype=np.uint8)
        
        # Convert PIL image to numpy
        sketch_array = np.array(sketch_input)
        
        # Convert to grayscale if needed
        if len(sketch_array.shape) == 3:
            sketch_array = cv2.cvtColor(sketch_array, cv2.COLOR_RGB2GRAY)
        
        # Resize to 256x256
        sketch_array = cv2.resize(sketch_array, (256, 256))
        
        # Ensure binary (0 or 255)
        sketch_array = (sketch_array > 128).astype(np.uint8) * 255
        
        return sketch_array
    
    def generate_image(
        self,
        sketch_input,
        prompt: str,
        text_guidance: float,
        sketch_guidance: float,
        num_steps: int,
        seed: int,
        show_attention: bool,
        show_timeline: bool,
    ) -> Tuple[Image.Image, Optional[Image.Image], Optional[Image.Image]]:
        """
        Generate image from sketch and prompt.
        
        Returns:
            - Generated image
            - Attention visualization (if requested)
            - Denoising timeline (if requested)
        """
        
        # Process sketch
        sketch_array = self.process_sketch(sketch_input)
        
        # For demo purposes, create a placeholder result
        # In practice, this would use your trained pipeline
        
        if self.pipeline is None:
            # Return placeholder images
            placeholder_img = self.create_placeholder_image(prompt)
            attention_img = self.create_placeholder_attention() if show_attention else None
            timeline_img = self.create_placeholder_timeline() if show_timeline else None
            
            return placeholder_img, attention_img, timeline_img
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate image using pipeline
        result = self.pipeline(
            prompt=prompt,
            sketch=sketch_array,
            guidance_scale_text=text_guidance,
            guidance_scale_sketch=sketch_guidance,
            num_inference_steps=num_steps,
            return_attention_maps=show_attention,
            return_timeline=show_timeline,
        )
        
        generated_image = result.images[0]
        
        # Create attention visualization
        attention_img = None
        if show_attention and hasattr(result, 'attention_maps'):
            attention_img = self.attention_viz.create_attention_grid(
                result.attention_maps, prompt, generated_image
            )
        
        # Create timeline visualization
        timeline_img = None
        if show_timeline and hasattr(result, 'timeline_images'):
            timeline_img = self.create_timeline_grid(result.timeline_images)
        
        return generated_image, attention_img, timeline_img
    
    def create_placeholder_image(self, prompt: str) -> Image.Image:
        """Create a placeholder image for demo purposes."""
        img = Image.new('RGB', (256, 256), color='lightgray')
        draw = ImageDraw.Draw(img)
        
        # Add some text
        draw.text((10, 120), f"Generated: {prompt[:20]}...", fill='black')
        draw.text((10, 140), "Model not loaded", fill='red')
        
        return img
    
    def create_placeholder_attention(self) -> Image.Image:
        """Create placeholder attention visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle("Word Attention Heatmaps", fontsize=16)
        
        words = ["red", "rose", "flower", "garden", "bloom", "petals"]
        
        for i, (ax, word) in enumerate(zip(axes.flat, words)):
            # Create random heatmap
            heatmap = np.random.random((16, 16))
            im = ax.imshow(heatmap, cmap='hot', interpolation='bilinear')
            ax.set_title(f'"{word}"')
            ax.axis('off')
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        attention_img = Image.open(buf)
        plt.close()
        
        return attention_img
    
    def create_placeholder_timeline(self) -> Image.Image:
        """Create placeholder denoising timeline."""
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        fig.suptitle("Denoising Timeline", fontsize=16)
        
        timesteps = [1000, 750, 500, 250, 0]
        
        for ax, t in zip(axes, timesteps):
            # Create random noise that gets cleaner
            noise_level = t / 1000.0
            img = np.random.random((64, 64, 3)) * noise_level + 0.5
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            ax.set_title(f"t={t}")
            ax.axis('off')
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        timeline_img = Image.open(buf)
        plt.close()
        
        return timeline_img
    
    def create_timeline_grid(self, timeline_images: List[Image.Image]) -> Image.Image:
        """Create a grid showing denoising timeline."""
        num_images = len(timeline_images)
        fig, axes = plt.subplots(1, num_images, figsize=(3 * num_images, 3))
        
        if num_images == 1:
            axes = [axes]
        
        for ax, img in zip(axes, timeline_images):
            ax.imshow(img)
            ax.axis('off')
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        timeline_img = Image.open(buf)
        plt.close()
        
        return timeline_img
    
    def create_interface(self) -> gr.Interface:
        """Create Gradio interface."""
        
        with gr.Blocks(title="ScribbleDiffusion Demo") as demo:
            gr.Markdown("""
            # 🎨 ScribbleDiffusion
            
            Turn your sketches into beautiful images with text prompts!
            
            **Instructions:**
            1. Draw a sketch in the canvas below
            2. Enter a text description
            3. Adjust guidance scales and generation settings
            4. Click Generate to create your image
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Input")
                    
                    # Sketch canvas
                    sketch_input = gr.Sketchpad(
                        label="Draw your sketch here",
                        height=256,
                        width=256,
                    )
                    
                    # Text prompt
                    prompt_input = gr.Textbox(
                        label="Text Prompt",
                        placeholder="e.g., 'a red rose in a garden'",
                        value="a red rose",
                    )
                    
                    gr.Markdown("### Generation Settings")
                    
                    with gr.Row():
                        text_guidance = gr.Slider(
                            label="Text Guidance",
                            minimum=1.0,
                            maximum=20.0,
                            value=7.5,
                            step=0.5,
                        )
                        sketch_guidance = gr.Slider(
                            label="Sketch Guidance", 
                            minimum=0.0,
                            maximum=5.0,
                            value=1.5,
                            step=0.1,
                        )
                    
                    with gr.Row():
                        num_steps = gr.Slider(
                            label="Inference Steps",
                            minimum=10,
                            maximum=100,
                            value=50,
                            step=10,
                        )
                        seed = gr.Number(
                            label="Seed",
                            value=42,
                            precision=0,
                        )
                    
                    with gr.Row():
                        show_attention = gr.Checkbox(
                            label="Show Attention Maps",
                            value=True,
                        )
                        show_timeline = gr.Checkbox(
                            label="Show Denoising Timeline",
                            value=True,
                        )
                    
                    generate_btn = gr.Button("Generate Image", variant="primary")
                
                with gr.Column(scale=2):
                    gr.Markdown("### Generated Results")
                    
                    generated_image = gr.Image(
                        label="Generated Image",
                        height=256,
                        width=256,
                    )
                    
                    with gr.Tabs():
                        with gr.Tab("Attention Maps"):
                            attention_viz = gr.Image(
                                label="Word-to-Region Attention",
                                height=400,
                            )
                        
                        with gr.Tab("Denoising Timeline"):
                            timeline_viz = gr.Image(
                                label="Generation Process",
                                height=200,
                            )
            
            # Set up the generation function
            generate_btn.click(
                fn=self.generate_image,
                inputs=[
                    sketch_input,
                    prompt_input,
                    text_guidance,
                    sketch_guidance,
                    num_steps,
                    seed,
                    show_attention,
                    show_timeline,
                ],
                outputs=[
                    generated_image,
                    attention_viz,
                    timeline_viz,
                ],
            )
            
            # Example sketches and prompts
            gr.Markdown("""
            ### Example Prompts to Try:
            - "a red rose in a beautiful garden"
            - "a blue sports car on a highway"
            - "a cozy house with a chimney"
            - "a majestic tree in autumn"
            - "a cute cat sitting on a windowsill"
            """)
        
        return demo


def main():
    """Launch the demo application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ScribbleDiffusion Demo")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./outputs/final_model",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the demo on",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link",
    )
    
    args = parser.parse_args()
    
    # Initialize demo
    demo_app = ScribbleDiffusionDemo(args.model_path)
    interface = demo_app.create_interface()
    
    # Launch
    interface.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0",
    )


if __name__ == "__main__":
    main()
