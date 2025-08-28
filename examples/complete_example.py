"""
Example usage of ScribbleDiffusion pipeline.
Demonstrates key features and capabilities.
"""

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch

# Note: This is a complete example that would work with installed dependencies
# For now it serves as documentation of the intended API


def create_example_sketch():
    """Create a simple example sketch."""
    # Create a 256x256 white canvas
    img = Image.new('L', (256, 256), 255)
    draw = ImageDraw.Draw(img)
    
    # Draw a simple house sketch
    # House base
    draw.rectangle([80, 150, 180, 200], outline=0, width=2)
    # Roof
    draw.polygon([(70, 150), (130, 100), (190, 150)], outline=0, width=2)
    # Door
    draw.rectangle([110, 170, 130, 200], outline=0, width=2)
    # Windows
    draw.rectangle([90, 160, 105, 175], outline=0, width=2)
    draw.rectangle([155, 160, 170, 175], outline=0, width=2)
    
    return img


def example_basic_generation():
    """Example of basic sketch-to-image generation."""
    print("🎨 ScribbleDiffusion Example: Basic Generation")
    print("=" * 50)
    
    # This would be the actual usage with installed dependencies:
    """
    from src.inference.pipeline import ScribbleDiffusionPipeline
    
    # Load trained pipeline
    pipeline = ScribbleDiffusionPipeline.from_pretrained("./outputs/final_model")
    
    # Create or load sketch
    sketch = create_example_sketch()
    
    # Generate image
    result = pipeline(
        prompt="a cozy house with red roof and blue door",
        sketch=sketch,
        guidance_scale_text=7.5,
        guidance_scale_sketch=1.5,
        num_inference_steps=50,
    )
    
    generated_image = result.images[0]
    generated_image.save("example_output.png")
    """
    
    # For now, show the concept
    sketch = create_example_sketch()
    sketch.save("example_sketch.png")
    print("✅ Created example sketch: example_sketch.png")
    
    print("\nWith trained model, you would:")
    print("1. Load the pipeline")
    print("2. Pass sketch + text prompt")
    print("3. Get generated image")


def example_attention_visualization():
    """Example of attention visualization features."""
    print("\n🔍 ScribbleDiffusion Example: Attention Visualization")
    print("=" * 50)
    
    # This demonstrates the attention visualization API:
    """
    from src.utils.attention_viz import AttentionVisualizer
    
    # Generate with attention tracking
    result = pipeline(
        prompt="a red house with blue windows",
        sketch=sketch,
        return_attention_maps=True,
    )
    
    # Visualize attention
    visualizer = AttentionVisualizer()
    attention_grid = visualizer.create_attention_grid(
        attention_maps=result.attention_maps,
        prompt="a red house with blue windows",
        generated_image=result.images[0],
    )
    
    attention_grid.save("attention_visualization.png")
    """
    
    print("Features include:")
    print("- Word-to-region attention heatmaps")
    print("- Token importance over denoising steps")
    print("- Interactive attention exploration")


def example_dual_guidance():
    """Example of dual guidance control."""
    print("\n⚖️ ScribbleDiffusion Example: Dual Guidance Control")
    print("=" * 50)
    
    # This shows how to control text vs sketch influence:
    """
    # High text guidance, low sketch guidance = follows text more
    result_text_heavy = pipeline(
        prompt="a futuristic building",
        sketch=house_sketch,
        guidance_scale_text=10.0,
        guidance_scale_sketch=0.5,
    )
    
    # Low text guidance, high sketch guidance = follows sketch more
    result_sketch_heavy = pipeline(
        prompt="a futuristic building",
        sketch=house_sketch,
        guidance_scale_text=3.0,
        guidance_scale_sketch=3.0,
    )
    
    # Balanced guidance
    result_balanced = pipeline(
        prompt="a futuristic building",
        sketch=house_sketch,
        guidance_scale_text=7.5,
        guidance_scale_sketch=1.5,
    )
    """
    
    print("Dual guidance allows:")
    print("- Text-heavy: Creative interpretation of sketch")
    print("- Sketch-heavy: Strict adherence to sketch structure")
    print("- Balanced: Optimal compromise between both")


def example_timeline_visualization():
    """Example of denoising timeline visualization."""
    print("\n📹 ScribbleDiffusion Example: Timeline Visualization")
    print("=" * 50)
    
    # This shows the denoising process:
    """
    result = pipeline(
        prompt="a beautiful house",
        sketch=sketch,
        return_timeline=True,
        timeline_steps=[49, 37, 25, 12, 0],  # Key timesteps
    )
    
    # Create timeline visualization
    timeline_images = result.timeline_images
    fig, axes = plt.subplots(1, len(timeline_images), figsize=(15, 3))
    
    for i, (ax, img) in enumerate(zip(axes, timeline_images)):
        ax.imshow(img)
        ax.set_title(f"Step {[49, 37, 25, 12, 0][i]}")
        ax.axis('off')
    
    plt.suptitle("Denoising Timeline: From Noise to Image")
    plt.savefig("timeline_visualization.png")
    """
    
    print("Timeline shows:")
    print("- Initial random noise")
    print("- Gradual structure emergence")
    print("- Detail refinement")
    print("- Final high-quality result")


def example_evaluation():
    """Example of model evaluation."""
    print("\n📊 ScribbleDiffusion Example: Model Evaluation")
    print("=" * 50)
    
    # This shows evaluation capabilities:
    """
    from src.utils.evaluation import ScribbleEvaluator
    
    evaluator = ScribbleEvaluator()
    
    # Generate test images
    test_sketches = [sketch1, sketch2, sketch3]
    test_prompts = ["house", "car", "tree"]
    
    generated_images = []
    for sketch, prompt in zip(test_sketches, test_prompts):
        result = pipeline(prompt=prompt, sketch=sketch)
        generated_images.append(result.images[0])
    
    # Evaluate
    metrics = evaluator.evaluate_batch(
        generated_images=generated_images,
        reference_sketches=test_sketches,
        text_prompts=test_prompts,
    )
    
    print(f"Edge Fidelity: {metrics['edge_fidelity_mean']:.3f}")
    print(f"CLIP Score: {metrics['clip_score_mean']:.3f}")
    print(f"Quality Score: {metrics['perceptual_quality_mean']:.3f}")
    
    # Create evaluation report
    report = evaluator.create_evaluation_report()
    report.save("evaluation_report.png")
    """
    
    print("Evaluation includes:")
    print("- Edge fidelity (sketch adherence)")
    print("- CLIP score (text alignment)")
    print("- Perceptual quality metrics")
    print("- Comprehensive reporting")


def example_training_workflow():
    """Example of training workflow."""
    print("\n🏋️ ScribbleDiffusion Example: Training Workflow")
    print("=" * 50)
    
    # This shows the training process:
    """
    # 1. Prepare dataset
    from src.data.dataset import ScribbleDataset
    
    dataset = ScribbleDataset(
        config=config.data,
        tokenizer=tokenizer,
        split="train",
    )
    
    # 2. Initialize models
    from src.models.unet import SketchConditionedUNet
    from src.models.hint_encoder import HintEncoder
    
    unet = SketchConditionedUNet(**config.model.unet)
    hint_encoder = HintEncoder(**config.model.hint_encoder)
    
    # 3. Train
    python train.py --config configs/base.yaml
    
    # 4. Monitor with wandb
    # Training logs show:
    # - Loss curves
    # - Validation samples
    # - Attention visualizations
    # - Resource usage
    """
    
    print("Training workflow:")
    print("1. Dataset preparation with edge augmentation")
    print("2. Model initialization (U-Net + Hint Encoder)")
    print("3. Training with dual guidance and EMA")
    print("4. Validation with attention visualization")
    print("5. Automatic checkpointing and logging")


def main():
    """Run all examples."""
    print("🎨 ScribbleDiffusion Complete Example Suite")
    print("=" * 60)
    print("\nThis demonstrates the full capabilities of ScribbleDiffusion:")
    print("- Sketch + text conditioning")
    print("- Attention visualization")
    print("- Dual guidance control")
    print("- Timeline visualization")
    print("- Comprehensive evaluation")
    print("- End-to-end training")
    
    # Run examples
    example_basic_generation()
    example_attention_visualization()
    example_dual_guidance()
    example_timeline_visualization()
    example_evaluation()
    example_training_workflow()
    
    print("\n🎉 Example suite complete!")
    print("\nTo run with real models:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Train model: python train.py --config configs/base.yaml")
    print("3. Run inference: python app.py")
    print("\nSee README.md for full documentation.")


if __name__ == "__main__":
    main()
