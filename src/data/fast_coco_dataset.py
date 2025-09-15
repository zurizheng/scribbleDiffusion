"""
Fast COCO Dataset loader - bypasses slow edge detection for speed testing
"""

import torch
import torch.utils.data
import numpy as np
from PIL import Image
import random
import json
from pathlib import Path
from typing import Dict, List, Tuple
from torchvision import transforms


class FastCOCOScribbleDataset(torch.utils.data.Dataset):
    """Ultra-fast COCO dataset that uses pre-generated random sketches instead of edge detection."""
    
    def __init__(self, config, tokenizer, split="train", download=True):
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        
        # Setup paths
        self.data_root = Path(config.get("data_root", "./data"))
        self.coco_root = self.data_root / "coco"
        self.images_dir = self.coco_root / f"{split}2017"
        
        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        
        # Load a small subset of images for testing
        self.image_files = []
        if self.images_dir.exists():
            all_files = list(self.images_dir.glob("*.jpg"))[:config.get("limit_dataset_size", 100)]
            self.image_files = all_files
        
        if not self.image_files:
            # Create dummy data for testing
            print(f"⚠️  No COCO images found, creating {config.get('limit_dataset_size', 10)} dummy samples")
            self.image_files = [f"dummy_{i}.jpg" for i in range(config.get("limit_dataset_size", 10))]
            self.use_dummy = True
        else:
            self.use_dummy = False
            
        print(f"FastCOCO dataset loaded: {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample - optimized for speed."""
        try:
            if self.use_dummy:
                # Generate dummy data instantly
                image_tensor = torch.randn(3, self.config.image_size, self.config.image_size) * 0.5
                sketch_tensor = torch.rand(1, self.config.image_size, self.config.image_size) > 0.95  # Sparse random sketch
                sketch_tensor = sketch_tensor.float()
                caption = f"a test image number {idx}"
                
            else:
                # Load real image (fast)
                image_path = self.image_files[idx]
                image = Image.open(image_path).convert("RGB")
                image_tensor = self.image_transform(image)
                
                # Generate random sketch instead of edge detection (ultra-fast)
                h, w = self.config.image_size, self.config.image_size
                sketch_tensor = torch.rand(1, h, w) > 0.95  # 5% random pixels as "edges"
                sketch_tensor = sketch_tensor.float()
                
                caption = f"an image from COCO dataset"
            
            # Tokenize caption (fast)
            token_data = self.tokenizer(
                caption,
                max_length=77,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            return {
                "images": image_tensor,
                "sketches": sketch_tensor,
                "input_ids": token_data["input_ids"].squeeze(0),
                "attention_mask": token_data["attention_mask"].squeeze(0),
                "caption": caption,
                "image_id": idx,
            }
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return dummy sample
            return {
                "images": torch.randn(3, self.config.image_size, self.config.image_size) * 0.5,
                "sketches": torch.rand(1, self.config.image_size, self.config.image_size) > 0.95,
                "input_ids": torch.zeros(77, dtype=torch.long),
                "attention_mask": torch.ones(77, dtype=torch.long),
                "caption": "dummy image",
                "image_id": idx,
            }