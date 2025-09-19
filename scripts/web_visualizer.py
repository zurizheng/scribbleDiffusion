#!/usr/bin/env python3
"""
ScribbleDiffusion Web Visualizer
Interactive web interface using Gradio for creating sketches and generating images
"""

import sys
import os
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import cv2
import time
import threading
import os
from pathlib import Path

# Import our pipeline
from scripts.fixed_inference import FixedScribblePipeline

# Import enhanced attention visualization
from src.utils.attention_viz import AttentionVisualizer

class ScribbleDiffusionApp:
    def __init__(self):
        self.pipeline = None
        self.pipeline_loading = False
        self.gallery_dir = Path("gallery")
        self.gallery_dir.mkdir(exist_ok=True)
        
        # Initialize pipeline and attention visualizer
        self.pipeline = FixedScribblePipeline()
        self.enhanced_attention_viz = AttentionVisualizer(gallery_root='gallery')
        
        # Load pipeline in background
        self.load_pipeline_async()
    
    def load_pipeline_async(self):
        """Load pipeline in background"""
        if self.pipeline_loading:
            return
        
        self.pipeline_loading = True
        
        def load():
            try:
                print("🚀 Loading ScribbleDiffusion pipeline...")
                self.pipeline = FixedScribblePipeline(force_cpu=False)
                print("✅ Pipeline loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading pipeline: {e}")
                self.pipeline = None
            finally:
                self.pipeline_loading = False
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def preprocess_sketch(self, sketch_img):
        """Convert sketch image to proper format for generation"""
        if sketch_img is None:
            return None
        
        # Handle Gradio Sketchpad output (which can be a dict with image data)
        if isinstance(sketch_img, dict):
            # Gradio Sketchpad returns dict with 'image' or 'composite' key
            if 'composite' in sketch_img:
                sketch_img = sketch_img['composite']
            elif 'image' in sketch_img:
                sketch_img = sketch_img['image']
            else:
                # If it's a dict but no expected keys, return None
                return None
        
        # Convert to PIL if needed
        if isinstance(sketch_img, np.ndarray):
            sketch_img = Image.fromarray(sketch_img)
        elif isinstance(sketch_img, str):
            # If it's a file path, load it
            sketch_img = Image.open(sketch_img)
        
        # Ensure we have a PIL Image now
        if not isinstance(sketch_img, Image.Image):
            return None
        
        # Convert to grayscale
        sketch_gray = sketch_img.convert("L")
        
        # Resize to 512x512
        sketch_gray = sketch_gray.resize((512, 512))
        
        # Check if there's actual drawing (not just white)
        sketch_array = np.array(sketch_gray)
        if np.all(sketch_array > 240):  # Mostly white
            return None
        
        # Apply edge detection for better sketch processing
        sketch_cv = cv2.Canny(sketch_array, 50, 150)
        
        return Image.fromarray(sketch_cv)
    
    def auto_save_to_gallery(self, image, prompt):
        """Save generated image to gallery"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_prompt = safe_prompt.replace(' ', '_')[:30]
        
        filename = f"{timestamp}_{safe_prompt}.png"
        filepath = self.gallery_dir / filename
        
        image.save(filepath)
        return str(filepath)
    
    def generate_image(self, sketch, prompt, steps, guidance_scale, seed, show_heatmaps=True, progress=gr.Progress()):
        """Generate image from sketch and prompt with enhanced attention analysis"""
        
        # Check if pipeline is ready
        if self.pipeline is None:
            if self.pipeline_loading:
                return None, None, None, "🔄 Pipeline is still loading... Please wait and try again."
            else:
                return None, None, None, "❌ Pipeline failed to load. Please check the console for errors."
        
        # Validate inputs
        if sketch is None:
            return None, None, None, "⚠️ Please draw a sketch first!"
        
        if not prompt.strip():
            return None, None, None, "⚠️ Please enter a prompt!"
        
        try:
            progress(0.1, desc="Processing sketch...")
            
            # Process the sketch
            processed_sketch = self.preprocess_sketch(sketch)
            if processed_sketch is None:
                return None, None, None, "⚠️ No drawing detected. Please draw something on the canvas!"
            
            progress(0.2, desc="Starting generation...")
            
            # Set seed
            actual_seed = None if seed == -1 else seed
            
            # Generate image with attention maps if requested
            result = self.pipeline.generate(
                prompt=prompt,
                sketch_path=processed_sketch,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                seed=actual_seed,
                return_attention_maps=show_heatmaps
            )
            
            progress(0.7, desc="Creating attention analysis...")
            
            # Create enhanced attention analysis if requested
            attention_heatmap = None
            evolution_gif = None
            
            if show_heatmaps and hasattr(result, 'attention_maps') and result.attention_maps:
                try:
                    # Create organized folder for this generation
                    folder_path = self.enhanced_attention_viz.create_prompt_folder(prompt)
                    
                    # Create comprehensive attention analysis
                    analysis = self.enhanced_attention_viz.create_comprehensive_attention_analysis(
                        prompt, processed_sketch, result, folder_path
                    )
                    
                    # Return the evolution GIF and comparison chart
                    if 'evolution_gif' in analysis and os.path.exists(analysis['evolution_gif']):
                        evolution_gif = analysis['evolution_gif']
                    
                    if 'text_vs_sketch' in analysis and os.path.exists(analysis['text_vs_sketch']):
                        attention_heatmap = analysis['text_vs_sketch']
                    
                    print(f"✅ Enhanced attention analysis saved to: {os.path.basename(folder_path)}")
                    
                except Exception as e:
                    print(f"Warning: Could not create enhanced attention analysis: {e}")
                    import traceback
                    traceback.print_exc()
                    attention_heatmap = self.create_placeholder_heatmap()
            elif show_heatmaps:
                attention_heatmap = self.create_placeholder_heatmap()
            
            progress(0.9, desc="Saving to gallery...")
            
            # Auto-save main image to gallery
            main_image = result.image if hasattr(result, 'image') else result
            gallery_path = self.auto_save_to_gallery(main_image, prompt)
            
            progress(1.0, desc="Complete!")
            
            # Return main image, static heatmap, evolution gif, and status
            success_msg = f"✅ Generated successfully! Saved to: {gallery_path}"
            if evolution_gif:
                success_msg += f"\n🎬 Attention evolution analysis available in gallery!"
            
            return main_image, attention_heatmap, evolution_gif, success_msg
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, None, None, f"❌ Generation failed: {str(e)}"
    
    def create_placeholder_heatmap(self):
        """Create a placeholder heatmap when attention maps are not available"""
        # Create a simple placeholder image
        placeholder = Image.new('RGB', (512, 512), color=(240, 240, 240))
        draw = ImageDraw.Draw(placeholder)
        
        # Add text
        text_lines = [
            "Attention Heatmap",
            "",
            "Heatmaps show where the model",
            "focuses when processing your",
            "text prompt and sketch.",
            "",
            "Red areas = high attention",
            "Blue areas = low attention"
        ]
        
        y_start = 200
        for i, line in enumerate(text_lines):
            draw.text((50, y_start + i * 25), line, fill=(100, 100, 100))
        
        return placeholder
    
    def load_gallery_images(self):
        """Load images from gallery for display"""
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(self.gallery_dir.glob(ext))
        
        # Sort by modification time (newest first)
        image_files = sorted(image_files, key=lambda x: x.stat().st_mtime, reverse=True)
        
        return [str(path) for path in image_files[:20]]  # Show last 20 images
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        with gr.Blocks(title="ScribbleDiffusion Visualizer", theme=gr.themes.Soft()) as app:
            
            gr.Markdown("""
            # ScribbleDiffusion Visualizer
            
            Draw a sketch and enter a prompt to generate amazing images! The AI will use your sketch as a guide to create the image.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ✏️ Draw Your Sketch")
                    sketch_input = gr.Sketchpad(
                        label="Draw here",
                        height=512,
                        width=512
                    )
                    
                    clear_btn = gr.Button("🗑️ Clear Sketch", variant="secondary")
                    clear_btn.click(fn=lambda: None, outputs=sketch_input)
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🖼️ Generated Image")
                    output_image = gr.Image(
                        label="Generated Result",
                        height=512,
                        width=512,
                        type="pil"
                    )
                    
                    status_text = gr.Textbox(
                        label="Status",
                        value="Ready to generate!",
                        interactive=False,
                        lines=2
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🔥 Attention Analysis")
                    attention_image = gr.Image(
                        label="Text vs Sketch Attention",
                        height=400,
                        width=512,
                        type="pil",
                        value=None
                    )
                    
                    evolution_gif = gr.Image(
                        label="🎬 Denoising Evolution",
                        height=300,
                        width=512,
                        type="pil",
                        value=None
                    )
                    
                    gr.Markdown("""
                    **Attention Analysis:**
                    - � **Top image**: Text vs Sketch token attention
                    - 🎬 **Bottom image**: Evolution during denoising (noise level → clean)
                    - � Red areas: High attention (model focuses here)
                    - Shows which sketch features drive generation
                    """)
            
            with gr.Row():
                with gr.Column():
                    prompt_input = gr.Textbox(
                        label="📝 Prompt",
                        placeholder="Describe what you want to generate (e.g., 'a red apple on a wooden table')",
                        value="a beautiful red apple",
                        lines=2
                    )
                
                with gr.Column():
                    generate_btn = gr.Button("🎨 Generate Image", variant="primary", scale=2)
            
            # Advanced settings
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                with gr.Row():
                    steps_slider = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=20,
                        step=1,
                        label="Inference Steps (more = better quality, slower)"
                    )
                    
                    guidance_slider = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale (how closely to follow prompt)"
                    )
                    
                    seed_input = gr.Number(
                        label="Seed (-1 for random)",
                        value=-1,
                        precision=0
                    )
                
                with gr.Row():
                    show_heatmaps = gr.Checkbox(
                        label="🔥 Show Attention Heatmaps",
                        value=True,
                        info="Generate visualization showing where the model focuses attention"
                    )
            
            # Gallery section
            with gr.Accordion("🖼️ Gallery", open=False):
                gr.Markdown("### Recent Generations")
                gallery = gr.Gallery(
                    label="Generated Images",
                    show_label=False,
                    elem_id="gallery",
                    columns=4,
                    rows=2,
                    height="auto"
                )
                
                refresh_gallery_btn = gr.Button("🔄 Refresh Gallery")
                refresh_gallery_btn.click(
                    fn=self.load_gallery_images,
                    outputs=gallery
                )
            
            # Connect the generate button
            generate_btn.click(
                fn=self.generate_image,
                inputs=[
                    sketch_input,
                    prompt_input,
                    steps_slider,
                    guidance_slider,
                    seed_input,
                    show_heatmaps
                ],
                outputs=[output_image, attention_image, evolution_gif, status_text]
            )
            
            # Auto-refresh gallery on page load
            app.load(fn=self.load_gallery_images, outputs=gallery)
            
            # Examples
            gr.Markdown("""
            ### 💡 Tips for better results:
            - Draw clear, simple outlines
            - Use descriptive prompts
            - Try different guidance scales (7.5 is usually good)
            - More inference steps = better quality but slower generation
            """)
            
            gr.Examples(
                examples=[
                    ["a red apple on a wooden table"],
                    ["a beautiful yellow banana"],
                    ["a fresh orange fruit"],
                    ["a ripe strawberry with green leaves"],
                    ["a purple grape cluster"]
                ],
                inputs=prompt_input,
                label="Example Prompts"
            )
        
        return app

def main():
    """Launch the web visualizer"""
    
    print("🚀 Starting ScribbleDiffusion Web Visualizer...")
    
    app_instance = ScribbleDiffusionApp()
    interface = app_instance.create_interface()
    
    # Launch with public access for remote use
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True if you want a public Gradio link
        debug=True,
        show_error=True
    )

if __name__ == "__main__":
    main()