#!/usr/bin/env python3
"""
Validation script for ScribbleDiffusion using pretrained SD 1.5 components
with zero-initialized HintEncoder for testing the pipeline.
"""

import torch
import yaml
import numpy as np
from PIL import Image, ImageDraw
import cv2
from pathlib import Path
from types import SimpleNamespace

from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from src.models.unet import SketchConditionedUNet
from src.models.hint_encoder import HintEncoder
from src.inference.pipeline import ScribbleDiffusionPipeline

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d

def create_test_sketch():
    """Create a simple test sketch."""
    # Create a simple sketch: circle and rectangle
    img = Image.new('RGB', (512, 512), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw circle
    draw.ellipse([150, 150, 350, 350], outline='black', width=3)
    # Draw rectangle  
    draw.rectangle([200, 400, 312, 450], outline='black', width=3)
    
    return img

def sketch_to_edge_map(sketch_img):
    """Convert sketch to binary edge map."""
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(sketch_img), cv2.COLOR_RGB2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Convert to PIL
    edge_img = Image.fromarray(edges)
    
    return edge_img

def main():
    print("🎨 ScribbleDiffusion Validation Test")
    print("=" * 50)
    
    # Load config
    with open('configs/validation.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config = dict_to_namespace(config)
    
    device = torch.device('cpu')  # Force CPU for 4GB VRAM compatibility 
    print(f"Using device: {device} (CPU mode for 4GB VRAM)")
    
    try:
        print("\n1. Loading pretrained components...")
        
        # Load pretrained VAE
        vae = AutoencoderKL.from_pretrained(
            config.model.vae.model_name,
            subfolder=config.model.vae.subfolder,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
        ).to(device)
        print("   ✅ VAE loaded")
        
        # Load pretrained text encoder
        text_encoder = CLIPTextModel.from_pretrained(
            config.model.text_encoder.model_name,
            subfolder=config.model.text_encoder.subfolder,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
        ).to(device)
        print("   ✅ Text encoder loaded")
        
        # Load tokenizer
        tokenizer = CLIPTokenizer.from_pretrained(
            config.model.text_encoder.model_name,
            subfolder="tokenizer"
        )
        print("   ✅ Tokenizer loaded")
        
        # Load pretrained UNet and convert to our custom UNet
        print("   Loading pretrained UNet...")
        from diffusers import UNet2DConditionModel
        pretrained_unet = UNet2DConditionModel.from_pretrained(
            config.model.unet.model_name,
            subfolder=config.model.unet.subfolder,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
        )
        
        # Create our custom UNet with matching architecture
        unet = SketchConditionedUNet(
            in_channels=4,
            out_channels=4,
            model_channels=320,  # SD 1.5 standard
            attention_resolutions=[2, 4, 8],
            num_res_blocks=2,
            channel_mult=[1, 2, 4, 4],
            num_heads=8,
            use_spatial_transformer=True,
            transformer_depth=1,
            context_dim=768,
            use_checkpoint=False  # Disable for validation
        ).to(device)
        
        # Copy pretrained weights to our custom UNet
        print("   Copying pretrained weights...")
        unet.load_state_dict(pretrained_unet.state_dict(), strict=False)
        del pretrained_unet  # Free memory
        print("   ✅ Custom UNet loaded with pretrained weights")
        
        # Load scheduler
        scheduler = DDIMScheduler.from_pretrained(
            config.model.unet.model_name,
            subfolder="scheduler"
        )
        print("   ✅ Scheduler loaded")
        
        print("\n2. Initializing HintEncoder...")
        
        # For pretrained SD 1.5 UNet, we need to match the standard channel progression
        # SD 1.5 UNet: 320 base channels with [1,2,4,4] multipliers
        # At injection points: [320, 320, 640, 1280] channels expected
        hint_encoder = HintEncoder(
            in_channels=1,
            hint_channels=[16, 32, 96, 256],  # Our encoder channels
            injection_layers=[0, 1, 2, 3],
            injection_method="add",
            unet_channels=[320, 320, 640, 1280]  # Match SD 1.5 UNet
        ).to(device)
        
        # Zero-initialize for ControlNet-style behavior
        for param in hint_encoder.parameters():
            param.data.zero_()
        
        print("   ✅ HintEncoder initialized (zero weights)")
        
        print("\n3. Creating test data...")
        
        # Create test sketch
        sketch = create_test_sketch()
        sketch.save("test_sketch.png")
        print("   ✅ Test sketch created: test_sketch.png")
        
        # Convert to edge map
        edge_map = sketch_to_edge_map(sketch)
        edge_map.save("test_edge_map.png")
        print("   ✅ Edge map created: test_edge_map.png")
        
        print("\n4. Testing inference pipeline...")
        
        # Create pipeline
        pipeline = ScribbleDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            hint_encoder=hint_encoder,
            scheduler=scheduler
        )
        
        # Test prompts
        prompts = [
            "a beautiful landscape painting",
            "a cute cat drawing", 
            "architectural blueprint",
        ]
        
        output_dir = Path("validation_outputs")
        output_dir.mkdir(exist_ok=True)
        
        for i, prompt in enumerate(prompts):
            print(f"   Generating image {i+1}: '{prompt}'")
            
            # Generate image
            with torch.no_grad():
                image = pipeline(
                    prompt=prompt,
                    sketch=edge_map,
                    guidance_scale_text=config.validation.guidance_scale_text,
                    guidance_scale_sketch=config.validation.guidance_scale_sketch,
                    num_inference_steps=config.validation.num_inference_steps,
                    height=512,
                    width=512
                ).images[0]
            
            # Save result
            output_path = output_dir / f"result_{i+1:02d}_{prompt.replace(' ', '_')[:20]}.png"
            image.save(output_path)
            print(f"     ✅ Saved: {output_path}")
        
        print("\n🎉 Validation completed successfully!")
        print("\nResults:")
        print(f"   📁 Output directory: {output_dir}")
        print(f"   🖼️  Generated images: {len(prompts)}")
        print("   📝 Test sketch: test_sketch.png")
        print("   📝 Edge map: test_edge_map.png")
        
        print("\n💡 Notes:")
        print("   • HintEncoder was zero-initialized, so sketch influence is minimal")
        print("   • This validates the pipeline works with pretrained SD 1.5 components")
        print("   • To get sketch conditioning, you'd need to train the HintEncoder")
        print("   • All components loaded successfully on your hardware!")
        
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✨ Validation pipeline is working! Ready for training or further development.")
    else:
        print("\n🔧 Check the errors above and fix any issues.")
