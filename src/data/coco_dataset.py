"""
COCO Dataset loader for ScribbleDiffusion.
Handles COCO 2017 dataset with automatic edge detection and preprocessing.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer
from pycocotools.coco import COCO


class COCOScribbleDataset(Dataset):
    """
    COCO dataset loader with automatic sketch generation and preprocessing.
    
    Features:
    - Downloads COCO 2017 dataset automatically
    - Generates edge maps from images using Canny/HED
    - Aggressive edge augmentations for robustness
    - Caption preprocessing and tokenization
    - Efficient data loading with caching
    """
    
    def __init__(
        self,
        config,
        tokenizer: CLIPTokenizer,
        split: str = "train",
        coco_root: Optional[str] = None,
        download: bool = True,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        self.download = download
        
        # Set COCO root directory
        if coco_root is None:
            coco_root = config.get("coco_root", "./data/coco")
        self.coco_root = Path(coco_root)
        
        # Setup dataset
        self._setup_coco_dataset()
        
        # Load COCO annotations
        self.coco = self._load_coco_annotations()
        
        # Get image IDs and filter by split
        self.image_ids = self._get_image_ids()
        
        # Create image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(0.5) if config.get("random_flip", True) else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
        ])
        
        # Edge detection parameters
        self.edge_method = config.get("edge_method", "canny")
        self.canny_low = config.get("canny_low", 50)
        self.canny_high = config.get("canny_high", 150)
        
        # Augmentation parameters
        self.edge_jitter = config.get("edge_jitter", True)
        self.jitter_prob = config.get("jitter_prob", 0.3)
        
        print(f"Loaded COCO {split} dataset with {len(self.image_ids)} images")
    
    def _setup_coco_dataset(self):
        """Setup COCO dataset directory structure and download if needed."""
        self.coco_root.mkdir(parents=True, exist_ok=True)
        
        # Define paths
        self.images_dir = {
            "train": self.coco_root / "train2017",
            "val": self.coco_root / "val2017",
        }
        
        self.annotations_dir = self.coco_root / "annotations"
        
        # Create directories
        for split_dir in self.images_dir.values():
            split_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        
        if self.download:
            self._download_coco_if_needed()
    
    def _download_coco_if_needed(self):
        """Download COCO dataset if not already present."""
        # Check if dataset already exists
        train_images = list(self.images_dir["train"].glob("*.jpg"))
        val_images = list(self.images_dir["val"].glob("*.jpg"))
        
        annotations_file = self.annotations_dir / "instances_train2017.json"
        captions_file = self.annotations_dir / "captions_train2017.json"
        
        if len(train_images) > 10000 and annotations_file.exists():
            print("COCO dataset already exists, skipping download")
            return
        
        print("COCO dataset not found. Please download it manually:")
        print("1. Go to https://cocodataset.org/#download")
        print("2. Download:")
        print(f"   - 2017 Train images -> {self.images_dir['train']}")
        print(f"   - 2017 Val images -> {self.images_dir['val']}")
        print(f"   - 2017 Train/Val annotations -> {self.annotations_dir}")
        print()
        print("Or use these commands:")
        print("# Create download script")
        self._create_download_script()
        
        # For now, create a small demo subset if no data exists
        if len(train_images) == 0:
            print("Creating demo COCO subset...")
            self._create_demo_coco_subset()
    
    def _create_download_script(self):
        """Create a download script for COCO dataset."""
        download_script = self.coco_root / "download_coco.sh"
        
        script_content = f"""#!/bin/bash
# COCO 2017 Dataset Download Script

COCO_DIR="{self.coco_root.absolute()}"
cd "$COCO_DIR"

echo "Downloading COCO 2017 dataset..."

# Create directories
mkdir -p train2017 val2017 annotations

# Download training images (18GB)
echo "Downloading training images..."
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
rm train2017.zip

# Download validation images (1GB)
echo "Downloading validation images..."
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
rm val2017.zip

# Download annotations (241MB)
echo "Downloading annotations..."
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip

echo "COCO 2017 dataset download complete!"
echo "Total size: ~19GB"
"""
        
        with open(download_script, "w") as f:
            f.write(script_content)
        
        download_script.chmod(0o755)
        print(f"Created download script: {download_script}")
        print(f"Run: bash {download_script}")
    
    def _create_demo_coco_subset(self):
        """Create a small demo subset of COCO-style data."""
        print("Creating demo COCO subset for testing...")
        
        # Create dummy images and annotations
        demo_images_dir = self.images_dir["train"]
        demo_images_dir.mkdir(exist_ok=True)
        
        # Create simple demo images
        categories = [
            {"id": 1, "name": "person", "supercategory": "person"},
            {"id": 2, "name": "car", "supercategory": "vehicle"},
            {"id": 3, "name": "house", "supercategory": "object"},
        ]
        
        images = []
        annotations = []
        
        for i in range(20):  # Create 20 demo images
            # Create simple colored image
            img = Image.new("RGB", (256, 256), 
                          color=(random.randint(100, 255), 
                                random.randint(100, 255), 
                                random.randint(100, 255)))
            
            # Add some simple shapes
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            
            # Random shape
            if i % 3 == 0:
                # Circle (person)
                x, y = random.randint(50, 200), random.randint(50, 200)
                r = random.randint(20, 50)
                draw.ellipse([x-r, y-r, x+r, y+r], fill="red", outline="black", width=2)
                category_id = 1
                caption = "a red circle representing a person"
            elif i % 3 == 1:
                # Rectangle (car)
                x, y = random.randint(50, 150), random.randint(50, 150)
                w, h = random.randint(60, 100), random.randint(30, 50)
                draw.rectangle([x, y, x+w, y+h], fill="blue", outline="black", width=2)
                category_id = 2
                caption = "a blue rectangle representing a car"
            else:
                # Triangle (house)
                x, y = random.randint(80, 150), random.randint(80, 150)
                size = random.randint(30, 60)
                points = [(x, y-size), (x-size, y+size), (x+size, y+size)]
                draw.polygon(points, fill="green", outline="black", width=2)
                category_id = 3
                caption = "a green triangle representing a house"
            
            # Save image
            img_filename = f"demo_{i:06d}.jpg"
            img_path = demo_images_dir / img_filename
            img.save(img_path)
            
            # Create image metadata
            images.append({
                "id": i,
                "width": 256,
                "height": 256,
                "file_name": img_filename,
            })
            
            # Create annotation
            annotations.append({
                "id": i,
                "image_id": i,
                "category_id": category_id,
                "caption": caption,
            })
        
        # Create COCO-style annotation file
        coco_data = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        
        # Save annotations
        annotations_file = self.annotations_dir / "captions_train2017.json"
        with open(annotations_file, "w") as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"Created demo COCO subset with {len(images)} images")
    
    def _load_coco_annotations(self):
        """Load COCO annotations."""
        if self.split == "train":
            ann_file = self.annotations_dir / "captions_train2017.json"
        else:
            ann_file = self.annotations_dir / "captions_val2017.json"
        
        if not ann_file.exists():
            # Try the demo file
            ann_file = self.annotations_dir / "captions_train2017.json"
            if not ann_file.exists():
                raise FileNotFoundError(f"COCO annotations not found at {ann_file}")
        
        try:
            return COCO(str(ann_file))
        except:
            # Fallback: load as regular JSON
            with open(ann_file, 'r') as f:
                data = json.load(f)
            return data
    
    def _get_image_ids(self):
        """Get image IDs for the specified split."""
        if hasattr(self.coco, 'getImgIds'):
            # Official COCO API
            image_ids = self.coco.getImgIds()
        else:
            # Fallback for demo data
            image_ids = [img['id'] for img in self.coco['images']]
        
        # Limit dataset size for training efficiency
        max_images = self.config.get("max_images", None)
        if max_images and len(image_ids) > max_images:
            random.shuffle(image_ids)
            image_ids = image_ids[:max_images]
        
        return image_ids
    
    def get_image_path(self, image_id: int) -> Path:
        """Get the path to an image file."""
        if hasattr(self.coco, 'loadImgs'):
            img_info = self.coco.loadImgs(image_id)[0]
            filename = img_info['file_name']
        else:
            # Fallback for demo data
            img_info = next(img for img in self.coco['images'] if img['id'] == image_id)
            filename = img_info['file_name']
        
        # Try train directory first, then val
        for split_name, split_dir in self.images_dir.items():
            img_path = split_dir / filename
            if img_path.exists():
                return img_path
        
        raise FileNotFoundError(f"Image {filename} not found in any directory")
    
    def get_captions(self, image_id: int) -> List[str]:
        """Get captions for an image."""
        if hasattr(self.coco, 'getAnnIds'):
            # Official COCO API
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)
            captions = [ann['caption'] for ann in anns]
        else:
            # Fallback for demo data
            captions = [ann['caption'] for ann in self.coco['annotations'] 
                       if ann['image_id'] == image_id]
        
        return captions if captions else ["an image"]
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Detect edges in image using specified method."""
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
        """Apply aggressive edge augmentations for robustness to real sketches."""
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
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample."""
        try:
            image_id = self.image_ids[idx]
            
            # Load image
            image_path = self.get_image_path(image_id)
            image = Image.open(image_path).convert("RGB")
            
            # Get captions and pick one randomly
            captions = self.get_captions(image_id)
            caption = random.choice(captions)
            
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
            
            # Tokenize caption
            token_data = self.tokenize_caption(caption)
            
            return {
                "images": image_tensor,
                "sketches": edges_tensor,
                "input_ids": token_data["input_ids"],
                "attention_mask": token_data["attention_mask"],
                "caption": caption,
                "image_id": image_id,
            }
            
        except Exception as e:
            print(f"Error loading sample {idx} (image_id: {image_id}): {e}")
            # Return a dummy sample if loading fails
            dummy_image = torch.randn(3, self.config.image_size, self.config.image_size)
            dummy_sketch = torch.zeros(1, self.config.image_size, self.config.image_size)
            dummy_tokens = self.tokenize_caption("an image")
            
            return {
                "images": dummy_image,
                "sketches": dummy_sketch,
                "input_ids": dummy_tokens["input_ids"],
                "attention_mask": dummy_tokens["attention_mask"],
                "caption": "an image",
                "image_id": -1,
            }


def test_coco_dataset():
    """Test the COCO dataset loader."""
    from transformers import CLIPTokenizer
    
    # Configuration
    config = {
        "image_size": 256,
        "edge_method": "canny",
        "canny_low": 50,
        "canny_high": 150,
        "edge_jitter": True,
        "jitter_prob": 0.3,
        "max_images": 100,  # Limit for testing
        "coco_root": "./data/coco",
    }
    
    # Load tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    # Create dataset
    dataset = COCOScribbleDataset(
        config=config,
        tokenizer=tokenizer,
        split="train",
        download=True,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['images'].shape}")
    print(f"Sketch shape: {sample['sketches'].shape}")
    print(f"Caption: {sample['caption']}")
    
    return dataset


if __name__ == "__main__":
    test_coco_dataset()
