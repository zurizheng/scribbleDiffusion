#!/usr/bin/env python3
"""
Check image dimensions in fruit dataset
"""

import os
from PIL import Image
from pathlib import Path
from collections import defaultdict

def check_image_dimensions(data_dir):
    """Check dimensions of all images in the fruit dataset"""
    
    fruit_dir = Path(data_dir)
    if not fruit_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return
    
    dimensions = defaultdict(int)
    issues = []
    total_images = 0
    
    print(f"Checking images in: {fruit_dir}")
    print("=" * 50)
    
    # Check each fruit folder
    for folder in sorted(fruit_dir.iterdir()):
        if not folder.is_dir():
            continue
            
        print(f"\n📁 {folder.name}:")
        folder_images = 0
        folder_issues = []
        
        # Check each image in folder
        for image_file in folder.iterdir():
            if image_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
                
            try:
                with Image.open(image_file) as img:
                    width, height = img.size
                    dimensions[f"{width}x{height}"] += 1
                    total_images += 1
                    folder_images += 1
                    
                    # Check if not 256x256
                    if width != 256 or height != 256:
                        issue = f"  ⚠️  {image_file.name}: {width}x{height}"
                        folder_issues.append(issue)
                        issues.append(f"{folder.name}/{image_file.name}: {width}x{height}")
                        
            except Exception as e:
                issue = f"  ❌ {image_file.name}: Error - {e}"
                folder_issues.append(issue)
                issues.append(f"{folder.name}/{image_file.name}: Error - {e}")
        
        print(f"   Images: {folder_images}")
        if folder_issues:
            print(f"   Issues: {len(folder_issues)}")
            for issue in folder_issues[:5]:  # Show first 5 issues
                print(issue)
            if len(folder_issues) > 5:
                print(f"   ... and {len(folder_issues) - 5} more")
        else:
            print("   ✅ All images are 256x256")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    print(f"Total images: {total_images}")
    print(f"Issues found: {len(issues)}")
    
    # Dimension breakdown
    print("\n📏 Dimension breakdown:")
    for dim, count in sorted(dimensions.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_images) * 100
        status = "✅" if dim == "256x256" else "⚠️"
        print(f"   {status} {dim}: {count} images ({percentage:.1f}%)")
    
    # Show problematic images
    if issues:
        print(f"\n⚠️  Non-256x256 images ({len(issues)} total):")
        for issue in issues[:10]:  # Show first 10
            print(f"   {issue}")
        if len(issues) > 10:
            print(f"   ... and {len(issues) - 10} more")
            
        print(f"\n💡 Recommendation:")
        if len(issues) < total_images * 0.1:  # Less than 10% issues
            print("   Most images are correct size. Consider resizing the few problematic ones.")
        else:
            print("   Many images need resizing. Consider batch resizing all images.")
    else:
        print("\n✅ All images are 256x256! Perfect for training.")
    
    return dimensions, issues

def resize_images_to_256(data_dir, backup=True):
    """Resize all images to 256x256"""
    
    fruit_dir = Path(data_dir)
    backup_dir = fruit_dir.parent / "fruits_backup" if backup else None
    
    if backup and not backup_dir.exists():
        print(f"Creating backup at: {backup_dir}")
        import shutil
        shutil.copytree(fruit_dir, backup_dir)
        print("✅ Backup created")
    
    total_resized = 0
    
    for folder in fruit_dir.iterdir():
        if not folder.is_dir():
            continue
            
        print(f"Resizing images in {folder.name}...")
        folder_resized = 0
        
        for image_file in folder.iterdir():
            if image_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
                
            try:
                with Image.open(image_file) as img:
                    if img.size != (256, 256):
                        # Resize to 256x256
                        resized = img.resize((256, 256), Image.LANCZOS)
                        resized.save(image_file)
                        folder_resized += 1
                        total_resized += 1
                        
            except Exception as e:
                print(f"❌ Error resizing {image_file}: {e}")
        
        print(f"   Resized {folder_resized} images")
    
    print(f"\n✅ Resized {total_resized} images to 256x256")

if __name__ == "__main__":
    import sys
    
    data_dir = "data/fruits" if len(sys.argv) < 2 else sys.argv[1]
    
    print("🔍 Fruit Dataset Image Dimension Checker")
    print("=" * 50)
    
    # Check dimensions
    dimensions, issues = check_image_dimensions(data_dir)
    
    # Ask if user wants to resize
    if issues:
        print(f"\n❓ Would you like to resize all images to 256x256?")
        print("   This will:")
        print("   1. Create a backup of your original images")
        print("   2. Resize all images to exactly 256x256")
        print("   3. Overwrite the originals with resized versions")
        
        response = input("\nProceed with resizing? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            resize_images_to_256(data_dir, backup=True)
            print("\n🎉 All images are now 256x256!")
        else:
            print("Skipping resize. You can run this script again anytime.")