#!/usr/bin/env python3
"""
Pre-compute sketches for the fruit dataset to avoid on-the-fly edge detection during training
"""

import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

def create_sketch(image_path, output_path):
    """Convert image to sketch using edge detection and save"""
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        img_np = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Invert so lines are white on black background
        sketch = 255 - edges
        
        # Save as grayscale PNG
        sketch_pil = Image.fromarray(sketch, mode='L')
        sketch_pil.save(output_path, 'PNG')
        
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def precompute_sketches(dataset_dir, sketches_dir):
    """Pre-compute sketches for all images in the dataset"""
    dataset_path = Path(dataset_dir)
    sketches_path = Path(sketches_dir)
    
    print(f"🎨 Pre-computing sketches for dataset: {dataset_path}")
    print(f"📁 Saving sketches to: {sketches_path}")
    
    # Create sketches directory structure
    sketches_path.mkdir(exist_ok=True)
    
    total_processed = 0
    total_errors = 0
    
    # Process each fruit category folder
    for fruit_dir in dataset_path.iterdir():
        if not fruit_dir.is_dir() or fruit_dir.name.startswith('.'):
            continue
            
        print(f"\n📂 Processing {fruit_dir.name}...")
        
        # Create corresponding sketch directory
        sketch_fruit_dir = sketches_path / fruit_dir.name
        sketch_fruit_dir.mkdir(exist_ok=True)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in fruit_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"   No images found in {fruit_dir.name}")
            continue
            
        # Process images with progress bar
        processed = 0
        errors = 0
        
        for image_file in tqdm(image_files, desc=f"  {fruit_dir.name}"):
            # Create output path (change extension to .png)
            sketch_file = sketch_fruit_dir / f"{image_file.stem}.png"
            
            # Skip if sketch already exists
            if sketch_file.exists():
                processed += 1
                continue
                
            # Create sketch
            if create_sketch(image_file, sketch_file):
                processed += 1
            else:
                errors += 1
        
        print(f"   ✅ Processed: {processed}, ❌ Errors: {errors}")
        total_processed += processed
        total_errors += errors
    
    print(f"\n🎉 Sketch pre-computation complete!")
    print(f"   Total processed: {total_processed}")
    print(f"   Total errors: {total_errors}")
    print(f"   Success rate: {(total_processed / (total_processed + total_errors) * 100):.1f}%")
    
    return total_processed, total_errors

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Pre-compute sketches for fruit dataset")
    parser.add_argument("--dataset", default="my-fruit-dataset", 
                       help="Path to fruit dataset directory")
    parser.add_argument("--output", default="my-fruit-dataset-sketches", 
                       help="Output directory for pre-computed sketches")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"❌ Dataset directory not found: {args.dataset}")
        return 1
    
    processed, errors = precompute_sketches(args.dataset, args.output)
    
    if errors > 0:
        print(f"\n⚠️  Warning: {errors} images failed to process")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Update your dataset class to load sketches from: {args.output}")
    print(f"   2. Set create_sketches=False to use pre-computed sketches")
    
    return 0

if __name__ == "__main__":
    exit(main())