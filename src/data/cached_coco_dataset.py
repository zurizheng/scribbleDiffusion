"""
Cached COCO Dataset with pre-computed edge detection.
Caches edge detection results to disk for fast training.
"""

import torch
import torch.utils.data
import numpy as np
from PIL import Image
import random
import json
import cv2
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torchvision import transforms
from tqdm import tqdm


class CachedCOCOScribbleDataset(torch.utils.data.Dataset):
    """
    COCO dataset with cached edge detection for fast training.
    
    Cache structure:
    data/coco/edges_cache/
    ├── metadata.json          # Cache metadata and settings
    ├── train2017/            # Cached edges for training images
    │   ├── 000000000001.npy  # Edge detection results as numpy arrays
    │   └── ...
    └── val2017/              # Cached edges for validation images
        └── ...
    """
    
    def __init__(
        self,
        config,
        tokenizer,
        split="train",
        download=True,
        rebuild_cache=False,
        cache_batch_size=100,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        self.rebuild_cache = rebuild_cache
        self.cache_batch_size = cache_batch_size
        
        # Edge detection settings
        self.edge_method = config.get("edge_method", "canny")
        self.canny_low = config.get("canny_low", 50)
        self.canny_high = config.get("canny_high", 150)
        self.edge_jitter = config.get("edge_jitter", True)
        self.jitter_prob = config.get("jitter_prob", 0.5)
        self.image_size = config.get("image_size", 512)
        
        # Setup paths
        self.data_root = Path(config.get("data_root", "/workspace/scribbleDiffusion/data"))
        self.coco_root = self.data_root / "coco"
        self.images_dir = self.coco_root / f"{split}2017"
        self.cache_dir = self.coco_root / "edges_cache"
        self.split_cache_dir = self.cache_dir / f"{split}2017"
        self.metadata_file = self.cache_dir / "metadata.json"
        
        # Create cache directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.split_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        
        # Load image files
        self.image_files = []
        if self.images_dir.exists():
            all_files = list(self.images_dir.glob("*.jpg"))
            
            # Limit dataset size if specified
            limit = config.get("limit_dataset_size")
            if limit is not None:
                all_files = all_files[:limit]
                
            self.image_files = all_files
            print(f"Found {len(self.image_files)} images in {split} split")
        else:
            print(f"⚠️  No COCO images found at {self.images_dir}")
            
        # Load or create cache
        self._setup_cache()
        
        print(f"CachedCOCO dataset ready: {len(self.image_files)} images")
        if hasattr(self, 'cached_count'):
            print(f"Cache status: {self.cached_count}/{len(self.image_files)} images cached")
    
    def _setup_cache(self):
        """Setup edge detection cache."""
        # Check if cache needs rebuilding
        if self.rebuild_cache or not self._is_cache_valid():
            print("🔄 Building edge detection cache...")
            self._build_cache()
        else:
            print("✅ Using existing edge detection cache")
            
        # Count cached files
        self.cached_count = len(list(self.split_cache_dir.glob("*.npy")))
    
    def _get_cache_settings_hash(self) -> str:
        """Get hash of cache settings to detect when cache needs rebuilding."""
        settings = {
            'edge_method': self.edge_method,
            'canny_low': self.canny_low,
            'canny_high': self.canny_high,
            'image_size': self.image_size,
            'split': self.split,
        }
        settings_str = json.dumps(settings, sort_keys=True)
        return hashlib.md5(settings_str.encode()).hexdigest()
    
    def _is_cache_valid(self) -> bool:
        """Check if existing cache is valid."""
        if not self.metadata_file.exists():
            return False
            
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                
            # Check if settings match
            current_hash = self._get_cache_settings_hash()
            cached_hash = metadata.get('settings_hash', '')
            
            if current_hash != cached_hash:
                print(f"🔄 Cache settings changed, rebuilding...")
                return False
                
            # Check if all images are cached
            cached_files = set(metadata.get('cached_files', {}).get(self.split, []))
            current_files = {f.stem for f in self.image_files}
            
            if cached_files != current_files:
                missing = current_files - cached_files
                if missing:
                    print(f"🔄 Missing {len(missing)} cached images, rebuilding...")
                return False
                
            return True
            
        except Exception as e:
            print(f"⚠️  Error reading cache metadata: {e}")
            return False
    
    def _build_cache(self):
        """Build edge detection cache."""
        print(f"Caching edge detection for {len(self.image_files)} images...")
        
        cached_files = []
        
        # Process images in batches with progress bar
        for i in tqdm(range(0, len(self.image_files), self.cache_batch_size), 
                     desc="Caching edges", unit="batch"):
            batch_files = self.image_files[i:i + self.cache_batch_size]
            
            for image_file in batch_files:
                try:
                    # Load image
                    image = Image.open(image_file).convert("RGB")
                    image_np = np.array(image)
                    
                    # Detect edges
                    edges = self._detect_edges(image_np)
                    
                    # Save to cache
                    cache_file = self.split_cache_dir / f"{image_file.stem}.npy"
                    np.save(cache_file, edges)
                    
                    cached_files.append(image_file.stem)
                    
                except Exception as e:
                    print(f"❌ Error caching {image_file.name}: {e}")
                    continue
        
        # Save metadata
        metadata = {
            'settings_hash': self._get_cache_settings_hash(),
            'cached_files': {self.split: cached_files},
            'cache_settings': {
                'edge_method': self.edge_method,
                'canny_low': self.canny_low,
                'canny_high': self.canny_high,
                'image_size': self.image_size,
            },
            'created_at': str(torch.utils.data.get_worker_info() or 'main'),
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"✅ Cache built: {len(cached_files)} images cached")
    
    def _detect_edges(self, image: np.ndarray) -> np.ndarray:
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
    
    def _augment_edges(self, edges: np.ndarray) -> np.ndarray:
        """Apply edge augmentations for robustness."""
        if not self.edge_jitter or random.random() > self.jitter_prob:
            return edges
        
        # Dilation/Erosion for line thickness variation
        kernel_size = random.randint(1, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        if random.random() < 0.5:
            edges = cv2.dilate(edges, kernel, iterations=1)
        else:
            edges = cv2.erode(edges, kernel, iterations=1)
        
        # Add random gaps
        if random.random() < 0.2:
            gap_size = random.randint(1, 3)
            gap_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_size, gap_size))
            edges = cv2.erode(edges, gap_kernel, iterations=1)
        
        return edges
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample using cached edges."""
        try:
            image_file = self.image_files[idx]
            image_id = image_file.stem
            
            # Load image
            image = Image.open(image_file).convert("RGB")
            image_tensor = self.image_transform(image)
            
            # Load cached edges
            cache_file = self.split_cache_dir / f"{image_id}.npy"
            
            if cache_file.exists():
                # Load from cache (FAST!)
                edges = np.load(cache_file)
            else:
                # Fallback: compute on-the-fly (SLOW!)
                print(f"⚠️  Cache miss for {image_id}, computing edges on-the-fly")
                image_np = np.array(image)
                edges = self._detect_edges(image_np)
            
            # Apply augmentations
            edges = self._augment_edges(edges)
            
            # Convert edges to tensor
            edges_pil = Image.fromarray(edges).convert("L")
            edges_tensor = transforms.ToTensor()(edges_pil)
            
            # Resize edges to match expected size
            if edges_tensor.shape[-1] != self.image_size:
                edges_tensor = transforms.Resize((self.image_size, self.image_size))(edges_tensor)
            
            # Generate dummy caption for now
            caption = f"an image from COCO dataset"
            
            # Tokenize caption
            token_data = self.tokenizer(
                caption,
                max_length=77,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            return {
                "images": image_tensor,
                "sketches": edges_tensor,
                "input_ids": token_data["input_ids"].squeeze(0),
                "attention_mask": token_data["attention_mask"].squeeze(0),
                "caption": caption,
                "image_id": image_id,
            }
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return dummy sample
            return {
                "images": torch.randn(3, self.image_size, self.image_size) * 0.5,
                "sketches": torch.rand(1, self.image_size, self.image_size) > 0.95,
                "input_ids": torch.zeros(77, dtype=torch.long),
                "attention_mask": torch.ones(77, dtype=torch.long),
                "caption": "dummy image",
                "image_id": str(idx),
            }
    
    def get_cache_info(self) -> Dict:
        """Get information about the cache."""
        cache_size_mb = sum(f.stat().st_size for f in self.split_cache_dir.glob("*.npy")) / (1024 * 1024)
        
        return {
            'cache_dir': str(self.cache_dir),
            'split_cache_dir': str(self.split_cache_dir),
            'total_images': len(self.image_files),
            'cached_images': self.cached_count,
            'cache_size_mb': cache_size_mb,
            'cache_coverage': self.cached_count / len(self.image_files) if self.image_files else 0,
        }
    
    def clear_cache(self):
        """Clear the edge detection cache."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            print(f"🗑️  Cleared cache: {self.cache_dir}")


def precompute_edges_script():
    """Standalone script to precompute edges for COCO dataset."""
    import argparse
    from transformers import CLIPTokenizer
    
    parser = argparse.ArgumentParser(description="Precompute COCO edge detection cache")
    parser.add_argument("--data_root", type=str, default="/workspace/scribbleDiffusion/data", help="Data root directory")
    parser.add_argument("--image_size", type=int, default=512, help="Image size")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild existing cache")
    args = parser.parse_args()
    
    # Dummy config
    config = {
        'data_root': args.data_root,
        'image_size': args.image_size,
        'limit_dataset_size': args.limit,
        'edge_method': 'canny',
        'canny_low': 50,
        'canny_high': 150,
    }
    
    # Dummy tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
    
    # Build cache for train split
    print("Building cache for training split...")
    train_dataset = CachedCOCOScribbleDataset(
        config=config,
        tokenizer=tokenizer,
        split="train",
        rebuild_cache=args.rebuild,
    )
    
    # Show cache info
    cache_info = train_dataset.get_cache_info()
    print(f"\n📊 Cache Info:")
    print(f"   Cache directory: {cache_info['cache_dir']}")
    print(f"   Total images: {cache_info['total_images']}")
    print(f"   Cached images: {cache_info['cached_images']}")
    print(f"   Cache size: {cache_info['cache_size_mb']:.1f} MB")
    print(f"   Coverage: {cache_info['cache_coverage']:.1%}")


if __name__ == "__main__":
    precompute_edges_script()