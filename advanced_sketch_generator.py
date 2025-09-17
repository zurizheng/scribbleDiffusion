#!/usr/bin/env python3
"""
Advanced sketch generation methods for more realistic sketches
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm
from pathlib import Path
import argparse

def method_canny_basic(image):
    """Basic Canny edge detection (current method)"""
    img_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    sketch = 255 - edges
    return Image.fromarray(sketch, mode='L')

def method_canny_enhanced(image):
    """Enhanced Canny with better preprocessing"""
    img_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Enhanced preprocessing
    # 1. Bilateral filter to preserve edges while smoothing
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 2. Adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(filtered)
    
    # 3. Multiple Canny with different thresholds, then combine
    edges1 = cv2.Canny(enhanced, 30, 100)
    edges2 = cv2.Canny(enhanced, 50, 150)
    edges3 = cv2.Canny(enhanced, 70, 200)
    
    # Combine edges
    combined = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))
    
    # Morphological operations to connect broken edges
    kernel = np.ones((2,2), np.uint8)
    connected = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    sketch = 255 - connected
    return Image.fromarray(sketch, mode='L')

def method_contour_based(image):
    """Contour-based sketch generation (more shape-aware)"""
    img_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Smooth the image
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Create binary image with adaptive threshold
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create sketch canvas
    sketch = np.ones_like(gray) * 255
    
    # Draw contours with varying thickness
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Filter small noise
            # Thickness based on contour size
            thickness = max(1, int(np.sqrt(area) / 50))
            cv2.drawContours(sketch, [contour], -1, 0, thickness)
    
    return Image.fromarray(sketch, mode='L')

def method_pencil_sketch(image):
    """Simulate pencil sketch style"""
    img_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Invert the image
    inverted = 255 - gray
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    
    # Invert the blurred image
    inverted_blur = 255 - blurred
    
    # Create pencil sketch by dividing gray by inverted blur
    sketch = cv2.divide(gray, inverted_blur, scale=256.0)
    
    # Enhance contrast
    sketch = cv2.convertScaleAbs(sketch, alpha=1.2, beta=10)
    
    return Image.fromarray(sketch, mode='L')

def method_stylized_edges(image):
    """Stylized edges with multiple techniques combined"""
    img_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Method 1: Laplacian edges
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    # Method 2: Sobel edges
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(sobel)
    
    # Method 3: Canny
    canny = cv2.Canny(gray, 50, 150)
    
    # Combine all methods
    combined = cv2.bitwise_or(laplacian, cv2.bitwise_or(sobel, canny))
    
    # Apply morphological operations for better connection
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    # Invert for white lines on black background
    sketch = 255 - combined
    
    return Image.fromarray(sketch, mode='L')

def method_artistic_sketch(image):
    """Most artistic and hand-drawn looking method"""
    img_np = np.array(image.convert('RGB'))
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Create a sketch using multiple edge detection techniques
    # 1. Edge detection with bilateral filtering
    bilateral = cv2.bilateralFilter(gray, 15, 75, 75)
    edges = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
    
    # 2. Apply morphological operations to make it look more hand-drawn
    kernel = np.ones((2,2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    
    # 3. Add some texture by combining with Canny
    canny = cv2.Canny(bilateral, 50, 150)
    
    # Combine adaptive threshold and canny
    sketch = cv2.bitwise_and(edges, 255 - canny)
    
    # 4. Add some randomness to make it look more hand-drawn
    noise = np.random.randint(0, 25, gray.shape).astype(np.uint8)
    sketch = cv2.addWeighted(sketch, 0.9, noise, 0.1, 0)
    
    return Image.fromarray(sketch, mode='L')

# Dictionary of available methods
SKETCH_METHODS = {
    'canny_basic': method_canny_basic,
    'canny_enhanced': method_canny_enhanced, 
    'contour': method_contour_based,
    'pencil': method_pencil_sketch,
    'stylized': method_stylized_edges,
    'artistic': method_artistic_sketch,
}

def generate_sample_sketches(image_path, output_dir):
    """Generate sample sketches using all methods for comparison"""
    image = Image.open(image_path).convert('RGB')
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    filename = Path(image_path).stem
    
    # Save original
    image.save(output_path / f"{filename}_original.jpg")
    
    # Generate sketches with all methods
    for method_name, method_func in SKETCH_METHODS.items():
        try:
            sketch = method_func(image)
            sketch.save(output_path / f"{filename}_{method_name}.png")
            print(f"✅ Generated {method_name} sketch")
        except Exception as e:
            print(f"❌ Error with {method_name}: {e}")

def regenerate_sketches_with_method(dataset_dir, sketches_dir, method_name):
    """Regenerate all sketches using a specific method"""
    if method_name not in SKETCH_METHODS:
        print(f"❌ Unknown method: {method_name}")
        print(f"Available methods: {list(SKETCH_METHODS.keys())}")
        return
    
    method_func = SKETCH_METHODS[method_name]
    dataset_path = Path(dataset_dir)
    sketches_path = Path(sketches_dir)
    
    print(f"🎨 Regenerating sketches using method: {method_name}")
    print(f"📁 Dataset: {dataset_path}")
    print(f"📁 Output: {sketches_path}")
    
    sketches_path.mkdir(exist_ok=True)
    
    total_processed = 0
    total_errors = 0
    
    # Process each fruit category
    for fruit_dir in dataset_path.iterdir():
        if not fruit_dir.is_dir() or fruit_dir.name.startswith('.'):
            continue
            
        print(f"\n📂 Processing {fruit_dir.name}...")
        
        sketch_fruit_dir = sketches_path / fruit_dir.name
        sketch_fruit_dir.mkdir(exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in fruit_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            continue
            
        processed = 0
        errors = 0
        
        for image_file in tqdm(image_files, desc=f"  {fruit_dir.name}"):
            sketch_file = sketch_fruit_dir / f"{image_file.stem}.png"
            
            try:
                image = Image.open(image_file).convert('RGB')
                sketch = method_func(image)
                sketch.save(sketch_file, 'PNG')
                processed += 1
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                errors += 1
        
        print(f"   ✅ Processed: {processed}, ❌ Errors: {errors}")
        total_processed += processed
        total_errors += errors
    
    print(f"\n🎉 Sketch regeneration complete!")
    print(f"   Method used: {method_name}")
    print(f"   Total processed: {total_processed}")
    print(f"   Total errors: {total_errors}")
    
    if total_errors == 0:
        print(f"✅ All sketches regenerated successfully!")

def main():
    parser = argparse.ArgumentParser(description="Advanced sketch generation for fruit dataset")
    parser.add_argument("--action", choices=['sample', 'regenerate'], required=True,
                       help="Action to perform")
    parser.add_argument("--image", help="Single image for sampling (required for sample action)")
    parser.add_argument("--dataset", default="my-fruit-dataset", 
                       help="Dataset directory")
    parser.add_argument("--sketches", default="my-fruit-dataset-sketches", 
                       help="Sketches output directory")
    parser.add_argument("--method", choices=list(SKETCH_METHODS.keys()), 
                       default="artistic", help="Sketch generation method")
    parser.add_argument("--output", default="sketch_samples", 
                       help="Output directory for samples")
    
    args = parser.parse_args()
    
    if args.action == 'sample':
        if not args.image:
            print("❌ --image required for sample action")
            return 1
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            return 1
        generate_sample_sketches(args.image, args.output)
        print(f"\n💡 Check {args.output} directory to compare different sketch methods")
        print(f"💡 To regenerate all sketches with your preferred method:")
        print(f"   python advanced_sketch_generator.py --action regenerate --method <method_name>")
        
    elif args.action == 'regenerate':
        if not os.path.exists(args.dataset):
            print(f"❌ Dataset not found: {args.dataset}")
            return 1
        regenerate_sketches_with_method(args.dataset, args.sketches, args.method)
        print(f"\n💡 Updated sketches can now be used for training!")
    
    return 0

if __name__ == "__main__":
    exit(main())