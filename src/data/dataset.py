"""
Dataset for ScribbleDiffusion training.
Handles image-sketch-text triplets with data augmentation.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer


class ScribbleDataset(Dataset):
    """
    Dataset that provides image-sketch-caption triplets for training.
    
    Features:
    - Automatic edge detection from images
    - Aggressive edge augmentations for robustness
    - Text preprocessing and tokenization
    - On-the-fly data augmentation
    """
    
    def __init__(
        self,
        config,
        tokenizer: CLIPTokenizer,
        split: str = "train",
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        
        # Load dataset (placeholder - you'll need to implement based on your data)
        self.image_paths, self.captions = self._load_data()
        
        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(0.5) if config.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
        ])
        
        # Edge detection parameters
        self.edge_method = config.edge_method
        self.canny_low = config.canny_low
        self.canny_high = config.canny_high
        
        # Augmentation parameters
        self.edge_jitter = config.get("edge_jitter", False)
        self.jitter_prob = config.get("jitter_prob", 0.3)
        
    def _load_data(self) -> Tuple[List[str], List[str]]:
        """Load image paths and captions based on dataset configuration."""
        
        if self.config.dataset_name == "coco" or self.config.get("dataset_type") == "coco":
            # Use COCO dataset
            return self._load_coco_data()
        else:
            # Load custom dataset
            return self._load_custom_data()
    
    def _load_coco_data(self) -> Tuple[List[str], List[str]]:
        """Load COCO dataset using the dedicated COCO loader."""
        try:
            from .coco_dataset import COCOScribbleDataset
            
            # Create COCO dataset instance to get the data
            coco_dataset = COCOScribbleDataset(
                config=self.config,
                tokenizer=self.tokenizer,
                split=self.split,
                download=self.config.get("download_coco", True),
            )
            
            # Extract image paths and captions
            image_paths = []
            captions = []
            
            for i, image_id in enumerate(coco_dataset.image_ids):
                try:
                    image_path = str(coco_dataset.get_image_path(image_id))
                    image_captions = coco_dataset.get_captions(image_id)
                    caption = image_captions[0] if image_captions else "an image"
                    
                    image_paths.append(image_path)
                    captions.append(caption)
                    
                except Exception as e:
                    print(f"Skipping image {image_id}: {e}")
                    continue
            
            print(f"Loaded {len(image_paths)} COCO images")
            return image_paths, captions
            
        except ImportError:
            print("COCO dataset dependencies not available. Install with: pip install pycocotools")
            print("Using demo dataset instead...")
            return self._load_custom_data()
    
    def _load_custom_data(self) -> Tuple[List[str], List[str]]:
        """Load custom dataset from directory structure."""
        # Expected structure:
        # data/
        #   images/
        #     image1.jpg
        #     image2.jpg
        #   captions.txt  # One caption per line, matching image order
        
        data_dir = Path(self.config.dataset_name)
        image_dir = data_dir / "images"
        caption_file = data_dir / "captions.txt"
        
        if not image_dir.exists() or not caption_file.exists():
            print(f"Warning: Custom dataset not found at {data_dir}")
            return [], []
        
        # Load image paths
        image_paths = sorted([str(p) for p in image_dir.glob("*.jpg")])
        image_paths.extend(sorted([str(p) for p in image_dir.glob("*.png")]))
        
        # Load captions
        captions = []
        if caption_file.exists():
            with open(caption_file, 'r') as f:
                captions = [line.strip() for line in f.readlines()]
        
        # Ensure same length
        min_len = min(len(image_paths), len(captions))
        return image_paths[:min_len], captions[:min_len]
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges in image using specified method.
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Binary edge map as numpy array
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        if self.edge_method == "canny":
            edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        elif self.edge_method == "sobel":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = (edges > edges.mean() + edges.std()).astype(np.uint8) * 255
        else:
            raise ValueError(f"Unsupported edge method: {self.edge_method}")
        
        return edges
    
    def augment_edges(self, edges: np.ndarray) -> np.ndarray:
        """
        Apply aggressive edge augmentations for robustness to real sketches.
        
        Args:
            edges: Binary edge map
            
        Returns:
            Augmented edge map
        """
        if not self.edge_jitter or random.random() > self.jitter_prob:
            return edges
        
        # Dilation/Erosion to simulate different line thickness
        kernel_size = random.randint(1, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        if random.random() < 0.5:
            # Dilate (thicker lines)
            edges = cv2.dilate(edges, kernel, iterations=1)
        else:
            # Erode (thinner lines)
            edges = cv2.erode(edges, kernel, iterations=1)
        
        # Add random gaps to simulate hand-drawn sketches
        if random.random() < 0.2:
            gap_size = random.randint(1, 3)
            gap_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_size, gap_size))
            mask = np.random.random(edges.shape) < 0.05  # 5% of pixels
            mask = cv2.dilate(mask.astype(np.uint8), gap_kernel)
            edges[mask > 0] = 0
        
        return edges
    
    def tokenize_caption(self, caption: str) -> Dict[str, torch.Tensor]:
        """Tokenize caption for CLIP text encoder."""
        # Truncate caption if too long
        max_length = self.config.get("max_length", 77)
        
        tokens = self.tokenizer(
            caption,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": tokens.input_ids.squeeze(0),
            "attention_mask": tokens.attention_mask.squeeze(0),
        }
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample."""
        try:
            # Load image
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert("RGB")
            
            # Convert to numpy for edge detection
            image_np = np.array(image)
            
            # Detect edges
            edges = self.detect_edges(image_np)
            
            # Augment edges
            edges = self.augment_edges(edges)
            
            # Apply image transforms
            image_tensor = self.image_transform(image)
            
            # Convert edges to tensor
            edges_pil = Image.fromarray(edges).convert("L")
            edges_tensor = transforms.ToTensor()(edges_pil)
            
            # Resize edges to match expected size
            if edges_tensor.shape[-1] != self.config.image_size:
                edges_tensor = transforms.Resize((self.config.image_size, self.config.image_size))(edges_tensor)
            
            # Get caption and tokenize
            caption = self.captions[idx] if idx < len(self.captions) else "a drawing"
            token_data = self.tokenize_caption(caption)
            
            return {
                "images": image_tensor,
                "sketches": edges_tensor,
                "input_ids": token_data["input_ids"],
                "attention_mask": token_data["attention_mask"],
                "caption": caption,
            }
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return a dummy sample if loading fails
            dummy_image = torch.randn(3, self.config.image_size, self.config.image_size)
            dummy_sketch = torch.zeros(1, self.config.image_size, self.config.image_size)
            dummy_tokens = self.tokenize_caption("a drawing")
            
            return {
                "images": dummy_image,
                "sketches": dummy_sketch,
                "input_ids": dummy_tokens["input_ids"],
                "attention_mask": dummy_tokens["attention_mask"],
                "caption": "a drawing",
            }


def create_demo_dataset(output_dir: str, num_samples: int = 100):
    """
    Create a small demo dataset for testing.
    Generates simple geometric shapes with corresponding captions.
    """
    import os
    from PIL import Image, ImageDraw
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    captions = []
    shapes = ["circle", "square", "triangle", "rectangle"]
    colors = ["red", "blue", "green", "yellow", "purple"]
    
    for i in range(num_samples):
        # Create simple geometric image
        img = Image.new("RGB", (256, 256), "white")
        draw = ImageDraw.Draw(img)
        
        shape = random.choice(shapes)
        color = random.choice(colors)
        
        # Draw shape
        if shape == "circle":
            x, y = random.randint(50, 150), random.randint(50, 150)
            r = random.randint(20, 50)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
            caption = f"a {color} {shape}"
        elif shape == "square":
            x, y = random.randint(50, 150), random.randint(50, 150)
            size = random.randint(40, 80)
            draw.rectangle([x, y, x+size, y+size], fill=color)
            caption = f"a {color} {shape}"
        # Add more shapes as needed...
        
        # Save image
        img_path = images_dir / f"sample_{i:04d}.png"
        img.save(img_path)
        captions.append(caption)
    
    # Save captions
    with open(output_dir / "captions.txt", "w") as f:
        for caption in captions:
            f.write(caption + "\n")
    
    print(f"Demo dataset created at {output_dir} with {num_samples} samples")
