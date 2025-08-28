#!/usr/bin/env python3
"""
Debug script to test module imports
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.resolve()
print(f"Project root: {project_root}")

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    
print(f"Python sys.path:")
for i, path in enumerate(sys.path[:5]):
    print(f"  [{i}] {path}")

print("\n🔍 Testing directory structure:")
src_path = project_root / "src"
print(f"src exists: {src_path.exists()}")
print(f"src is dir: {src_path.is_dir()}")

models_path = src_path / "models"
print(f"src/models exists: {models_path.exists()}")
print(f"src/models is dir: {models_path.is_dir()}")

data_path = src_path / "data"
print(f"src/data exists: {data_path.exists()}")
print(f"src/data is dir: {data_path.is_dir()}")

print("\n🔍 Testing __init__.py files:")
src_init = src_path / "__init__.py"
models_init = models_path / "__init__.py"
data_init = data_path / "__init__.py"

print(f"src/__init__.py exists: {src_init.exists()}")
print(f"src/models/__init__.py exists: {models_init.exists()}")
print(f"src/data/__init__.py exists: {data_init.exists()}")

print("\n🔍 Testing imports step by step:")

try:
    print("Importing src...")
    import src
    print("✅ src imported successfully")
except Exception as e:
    print(f"❌ src import failed: {e}")

try:
    print("Importing src.models...")
    import src.models
    print("✅ src.models imported successfully")
except Exception as e:
    print(f"❌ src.models import failed: {e}")

try:
    print("Importing src.data...")
    import src.data
    print("✅ src.data imported successfully")
except Exception as e:
    print(f"❌ src.data import failed: {e}")

try:
    print("Importing src.models.unet...")
    from src.models.unet import SketchConditionedUNet
    print("✅ SketchConditionedUNet imported successfully")
except Exception as e:
    print(f"❌ SketchConditionedUNet import failed: {e}")

try:
    print("Importing src.data.dataset...")
    from src.data.dataset import ScribbleDataset
    print("✅ ScribbleDataset imported successfully")
except Exception as e:
    print(f"❌ ScribbleDataset import failed: {e}")

print("\n🎯 Debug complete!")
