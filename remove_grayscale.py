#!/usr/bin/env python3
"""
Detect and remove black and white (grayscale) images from fruit dataset
"""

import os
from PIL import Image
from pathlib import Path
import numpy as np
import shutil

def is_grayscale(image_path, threshold=10):
    """
    Check if an image is grayscale (black and white)
    
    Args:
        image_path: Path to image file
        threshold: Variance threshold below which image is considered grayscale
    
    Returns:
        bool: True if image is grayscale
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Get RGB channels
            r, g, b = img.split()
            
            # Convert to numpy arrays
            r_arr = np.array(r)
            g_arr = np.array(g)
            b_arr = np.array(b)
            
            # Calculate variance between channels
            # If all channels are very similar, it's grayscale
            rg_var = np.var(r_arr - g_arr)
            rb_var = np.var(r_arr - b_arr)
            gb_var = np.var(g_arr - b_arr)
            
            # Average variance between channels
            avg_var = (rg_var + rb_var + gb_var) / 3
            
            return avg_var < threshold
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def detect_grayscale_images(data_dir, threshold=10):
    """Detect all grayscale images in the dataset"""
    
    fruit_dir = Path(data_dir)
    if not fruit_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return []
    
    grayscale_images = []
    total_images = 0
    
    print(f"🔍 Detecting grayscale images in: {fruit_dir}")
    print(f"Variance threshold: {threshold} (lower = more strict)")
    print("=" * 60)
    
    # Check each fruit folder
    for folder in sorted(fruit_dir.iterdir()):
        if not folder.is_dir():
            continue
            
        print(f"\n📁 {folder.name}:")
        folder_images = 0
        folder_grayscale = []
        
        # Check each image in folder
        for image_file in folder.iterdir():
            if image_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
                
            total_images += 1
            folder_images += 1
            
            if is_grayscale(image_file, threshold):
                folder_grayscale.append(image_file)
                grayscale_images.append(image_file)
        
        print(f"   Total images: {folder_images}")
        print(f"   Grayscale found: {len(folder_grayscale)}")
        
        # Show first few grayscale images
        if folder_grayscale:
            print("   Grayscale images:")
            for img in folder_grayscale[:5]:
                print(f"     📷 {img.name}")
            if len(folder_grayscale) > 5:
                print(f"     ... and {len(folder_grayscale) - 5} more")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DETECTION SUMMARY")
    print("=" * 60)
    print(f"Total images checked: {total_images}")
    print(f"Grayscale images found: {len(grayscale_images)}")
    print(f"Percentage grayscale: {(len(grayscale_images)/total_images)*100:.1f}%")
    
    return grayscale_images

def remove_grayscale_images(grayscale_images, backup=True):
    """Remove grayscale images with optional backup"""
    
    if not grayscale_images:
        print("No grayscale images to remove!")
        return
    
    # Create backup directory if requested
    if backup:
        backup_dir = Path("data/grayscale_backup")
        backup_dir.mkdir(exist_ok=True)
        print(f"📦 Creating backup at: {backup_dir}")
    
    removed_count = 0
    
    for img_path in grayscale_images:
        try:
            # Backup if requested
            if backup:
                backup_folder = backup_dir / img_path.parent.name
                backup_folder.mkdir(exist_ok=True)
                backup_path = backup_folder / img_path.name
                shutil.copy2(img_path, backup_path)
            
            # Remove original
            img_path.unlink()
            removed_count += 1
            
        except Exception as e:
            print(f"❌ Error removing {img_path}: {e}")
    
    print(f"✅ Removed {removed_count} grayscale images")
    if backup:
        print(f"📦 Backup saved to: data/grayscale_backup")

def show_sample_images(grayscale_images, num_samples=3):
    """Show sample grayscale images for user verification"""
    
    if not grayscale_images:
        return
    
    print(f"\n🖼️  Sample grayscale images found:")
    print("=" * 40)
    
    # Group by folder
    by_folder = {}
    for img in grayscale_images:
        folder_name = img.parent.name
        if folder_name not in by_folder:
            by_folder[folder_name] = []
        by_folder[folder_name].append(img)
    
    # Show samples from each folder
    for folder, images in by_folder.items():
        print(f"\n📁 {folder} ({len(images)} grayscale):")
        for img in images[:num_samples]:
            # Try to get some basic stats
            try:
                with Image.open(img) as pil_img:
                    if pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')
                    r, g, b = pil_img.split()
                    r_arr, g_arr, b_arr = np.array(r), np.array(g), np.array(b)
                    
                    # Check how similar the channels are
                    rg_var = np.var(r_arr - g_arr)
                    similarity = f"R-G variance: {rg_var:.1f}"
                    
                    print(f"   📷 {img.name} ({similarity})")
                    
            except Exception:
                print(f"   📷 {img.name}")
        
        if len(images) > num_samples:
            print(f"   ... and {len(images) - num_samples} more")

if __name__ == "__main__":
    import sys
    
    data_dir = "data/fruits" if len(sys.argv) < 2 else sys.argv[1]
    threshold = 10 if len(sys.argv) < 3 else float(sys.argv[2])
    
    print("🔍 Fruit Dataset Grayscale Image Detector")
    print("=" * 60)
    
    # Detect grayscale images
    grayscale_images = detect_grayscale_images(data_dir, threshold)
    
    if not grayscale_images:
        print("\n✅ No grayscale images found! All images appear to be color.")
        sys.exit(0)
    
    # Show samples for verification
    show_sample_images(grayscale_images)
    
    # Ask user what to do
    print(f"\n❓ Found {len(grayscale_images)} grayscale images.")
    print("Options:")
    print("1. Remove them (with backup)")
    print("2. Remove them (no backup)")
    print("3. Keep them")
    print("4. Adjust threshold and re-detect")
    
    while True:
        choice = input("\nEnter choice (1/2/3/4): ").strip()
        
        if choice == "1":
            remove_grayscale_images(grayscale_images, backup=True)
            print("\n🎉 Grayscale images removed! Dataset is now color-only.")
            break
        elif choice == "2":
            confirm = input("⚠️  Are you sure? This cannot be undone (y/n): ").lower()
            if confirm in ['y', 'yes']:
                remove_grayscale_images(grayscale_images, backup=False)
                print("\n🎉 Grayscale images removed! Dataset is now color-only.")
            break
        elif choice == "3":
            print("Keeping all images as-is.")
            break
        elif choice == "4":
            new_threshold = input(f"Enter new threshold (current: {threshold}): ").strip()
            try:
                threshold = float(new_threshold)
                print(f"Re-detecting with threshold {threshold}...")
                grayscale_images = detect_grayscale_images(data_dir, threshold)
                if grayscale_images:
                    show_sample_images(grayscale_images)
                else:
                    print("No grayscale images found with new threshold.")
                    break
            except ValueError:
                print("Invalid threshold. Please enter a number.")
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")