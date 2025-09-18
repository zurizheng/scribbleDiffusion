#!/usr/bin/env python3
"""
Upload ScribbleDiffusion model to Hugging Face Hub
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, Repository, create_repo
    from huggingface_hub.utils import RepositoryNotFoundError
except ImportError:
    print("❌ huggingface_hub not installed. Install with: pip install huggingface_hub")
    sys.exit(1)

def upload_to_hf(model_path, repo_name, private=False, token=None):
    """Upload model to Hugging Face Hub"""
    
    print(f"🤗 Uploading to Hugging Face: {repo_name}")
    
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model path not found: {model_path}")
        return False
    
    # Initialize HF API (token=None lets it auto-detect from huggingface-cli login)
    api = HfApi(token=token)
    
    # Create repository if it doesn't exist
    try:
        print(f"🔍 Checking if repository exists...")
        api.repo_info(repo_id=repo_name)
        print(f"✅ Repository {repo_name} found")
    except RepositoryNotFoundError:
        print(f"📝 Creating new repository: {repo_name}")
        create_repo(
            repo_id=repo_name,
            private=private,
            token=token,  # This will also use auto-detection if token is None
            repo_type="model",
            exist_ok=True
        )
        print(f"✅ Repository created")
    
    # Upload files
    try:
        print(f"📤 Uploading model files...")
        
        # Upload all files in the model directory
        for file_path in model_path.iterdir():
            if file_path.is_file():
                print(f"   Uploading {file_path.name}...")
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=file_path.name,
                    repo_id=repo_name,
                    token=token
                )
        
        print(f"✅ Upload completed!")
        print(f"🌐 Model available at: https://huggingface.co/{repo_name}")
        
        # Show usage example
        print(f"\n💡 Usage example:")
        print(f"```python")
        print(f"from huggingface_hub import hf_hub_download")
        print(f"")
        print(f"# Download UNet")
        print(f"unet_path = hf_hub_download('{repo_name}', 'unet.safetensors')")
        print(f"")
        print(f"# Download HintEncoder")
        print(f"hint_path = hf_hub_download('{repo_name}', 'hint_encoder.safetensors')")
        print(f"```")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Upload ScribbleDiffusion model to Hugging Face")
    parser.add_argument("--model_path", required=True, help="Path to exported model directory")
    parser.add_argument("--repo_name", required=True, help="HuggingFace repo name (username/model-name)")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    parser.add_argument("--token", help="HuggingFace token (or set HF_TOKEN env var)")
    
    args = parser.parse_args()
    
    # Get token from args or environment, or let HF Hub auto-detect
    token = args.token or os.getenv("HF_TOKEN")
    # Note: If no explicit token, HfApi() will auto-detect from huggingface-cli login
    
    success = upload_to_hf(args.model_path, args.repo_name, args.private, token)
    
    if success:
        print(f"\n🎉 Model successfully uploaded!")
        print(f"📦 Available at: https://huggingface.co/{args.repo_name}")
        print(f"📖 Update your README.md with the model link")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
