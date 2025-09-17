import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path

class FruitDataset(Dataset):
    def __init__(self, data_dir, transform=None, image_size=256, create_sketches=True, sketches_dir=None):
        """
        Dataset for fruit images with automatic fruit type detection from folder names
        
        Args:
            data_dir: Path to fruit dataset directory (e.g., 'data/fruits')
            transform: Optional transform to apply to images
            image_size: Size to resize images to
            create_sketches: Whether to generate sketch versions of images on-the-fly
            sketches_dir: Path to pre-computed sketches directory (if None, uses on-the-fly generation)
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.create_sketches = create_sketches
        self.sketches_dir = sketches_dir
        self.image_paths = []
        self.fruit_labels = []
        
        # Fruit type mappings for folder names like 'Apple_Good' -> 'apple'
        self.fruit_types = ['apple', 'banana', 'guava', 'lime', 'orange', 'pomegranate']
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
        
        # Sketch transform (no normalization for sketches)
        self.sketch_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        
        # Load dataset from folder structure
        self._load_folder_organized()
        
        print(f"Loaded {len(self.image_paths)} fruit images:")
        for fruit in self.fruit_types:
            count = self.fruit_labels.count(fruit)
            print(f"  {fruit}: {count} images")
        
    def _get_fruit_name_from_folder(self, folder_name):
        """Extract fruit name from folder like 'Apple_Good' -> 'apple'"""
        folder_lower = folder_name.lower()
        
        # Check each fruit type
        for fruit_name in self.fruit_types:
            if fruit_name in folder_lower:
                return fruit_name
                
        # If no direct match, return None
        return None
    
    def _load_folder_organized(self):
        """Load dataset organized by fruit folders"""
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Dataset directory not found: {self.data_dir}")
        
        # Get all subdirectories
        subdirs = [d for d in os.listdir(self.data_dir) 
                   if os.path.isdir(os.path.join(self.data_dir, d))]
        
        if not subdirs:
            raise ValueError(f"No subdirectories found in {self.data_dir}")
        
        # Process each fruit folder
        for folder_name in subdirs:
            fruit_name = self._get_fruit_name_from_folder(folder_name)
            
            if fruit_name is None:
                print(f"Warning: Could not identify fruit type for folder '{folder_name}', skipping")
                continue
            
            folder_path = os.path.join(self.data_dir, folder_name)
            
            # Get all image files in this folder
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
            for filename in os.listdir(folder_path):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    image_path = os.path.join(folder_path, filename)
                    self.image_paths.append(image_path)
                    self.fruit_labels.append(fruit_name)
    
    def _create_sketch(self, image):
        """Convert image to sketch using edge detection"""
        # Convert PIL to numpy
        img_np = np.array(image.convert('RGB'))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Invert so lines are white on black background
        sketch = 255 - edges
        
        # Convert back to PIL as grayscale (1 channel) since sketch encoder expects 1 channel
        sketch_pil = Image.fromarray(sketch).convert('L')  # 'L' mode for grayscale
        
        return sketch_pil
    
    def _load_precomputed_sketch(self, image_path):
        """Load pre-computed sketch from sketches directory"""
        # Convert image path to corresponding sketch path
        image_path_obj = Path(image_path)
        relative_path = image_path_obj.relative_to(self.data_dir)
        
        # Change extension to .png for sketch
        sketch_filename = relative_path.stem + '.png'
        sketch_path = Path(self.sketches_dir) / relative_path.parent / sketch_filename
        
        if not sketch_path.exists():
            # Fallback to on-the-fly generation if sketch not found
            print(f"Warning: Pre-computed sketch not found at {sketch_path}, generating on-the-fly")
            image = Image.open(image_path).convert('RGB')
            return self._create_sketch(image)
        
        # Load pre-computed sketch
        sketch = Image.open(sketch_path).convert('L')
        return sketch
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        fruit_label = self.fruit_labels[idx]
        
        try:
            # Load and process original image
            image = Image.open(image_path).convert('RGB')
            
            # Handle sketch loading
            if self.create_sketches:
                if self.sketches_dir:
                    # Load pre-computed sketch
                    sketch = self._load_precomputed_sketch(image_path)
                else:
                    # Create sketch on-the-fly (memory intensive)
                    sketch = self._create_sketch(image)
                    
                sketch_tensor = self.sketch_transform(sketch)
            else:
                sketch_tensor = None
            
            # Apply transforms
            image_tensor = self.transform(image)
            
            # Return dictionary with all data
            sample = {
                'image': image_tensor,
                'fruit_label': fruit_label,
                'text_prompt': f"a {fruit_label}",  # Simple text prompt
                'image_path': image_path
            }
            
            if sketch_tensor is not None:
                sample['sketch'] = sketch_tensor
            
            return sample
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a placeholder or skip
            raise e

def test_dataset():
    """Test the fruit dataset"""
    dataset = FruitDataset('data/fruits')
    
    if len(dataset) == 0:
        print("No images found in dataset!")
        return
    
    # Test loading a few samples
    print(f"\nTesting dataset with {len(dataset)} images...")
    
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"Sample {i}:")
        print(f"  Fruit: {sample['fruit_label']}")
        print(f"  Text prompt: {sample['text_prompt']}")
        print(f"  Image shape: {sample['image'].shape}")
        if 'sketch' in sample:
            print(f"  Sketch shape: {sample['sketch'].shape}")

if __name__ == "__main__":
    test_dataset()