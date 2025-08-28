# ScribbleDiffusion Model Deployment Guide

## Model Storage and Sharing Options

After training your ScribbleDiffusion model on RTX 3090, you have several options for sharing and deploying it:

### 🤗 Option 1: Hugging Face Hub (Recommended)
**Best for**: Public sharing, easy integration, community access

```bash
# Install Hugging Face Hub
pip install huggingface_hub

# Login to your HF account
huggingface-cli login

# Upload your trained model
python scripts/upload_to_hf.py --model_path outputs/rtx3090_training/final_model
```

**Advantages:**
- Free for public models
- Integrated with diffusers library
- Easy loading: `pipe = StableDiffusionPipeline.from_pretrained("your-username/scribble-diffusion")`
- Version control for models
- Community can easily use your model

### 🐙 Option 2: Git LFS (Large File Storage)
**Best for**: Version-controlled models, team collaboration

```bash
# Install Git LFS
git lfs install

# Track model files (update .gitattributes)
git lfs track "models/*.safetensors"
git lfs track "models/*.pth"

# Add and commit
git add models/scribble_diffusion_rtx3090.safetensors
git commit -m "Add trained ScribbleDiffusion model"
git push
```

**Advantages:**
- Integrated with Git workflow
- Version history for models
- Works with GitHub (100MB file limit without LFS)

### ☁️ Option 3: Cloud Storage Links
**Best for**: Private sharing, large models

```bash
# Google Drive, Dropbox, or AWS S3
# Create a download script for users
echo "wget https://drive.google.com/your-model-link -O models/scribble_diffusion.safetensors" > download_model.sh
```

### 🐳 Option 4: Docker with Model
**Best for**: Complete deployment solution

```dockerfile
# Dockerfile with trained model
FROM pytorch/pytorch:latest
COPY models/ /app/models/
COPY src/ /app/src/
RUN pip install diffusers transformers
CMD ["python", "inference.py"]
```

## Recommended Approach for ScribbleDiffusion

### Step 1: Prepare Model for Upload
```bash
# After training completes, save in standard format
python scripts/export_model.py \
    --checkpoint outputs/rtx3090_training/checkpoint-best.pth \
    --output models/scribble-diffusion-v1 \
    --format safetensors
```

### Step 2: Upload to Hugging Face
```bash
# Upload with proper model card
python scripts/upload_to_hf.py \
    --model_path models/scribble-diffusion-v1 \
    --repo_name "your-username/scribble-diffusion-rtx3090" \
    --private False
```

### Step 3: Update Repository
```bash
# Add model info to README
echo "## 🎨 Pre-trained Model" >> README.md
echo "Download: https://huggingface.co/your-username/scribble-diffusion-rtx3090" >> README.md

# Commit the change
git add README.md
git commit -m "Add link to trained model on Hugging Face"
git push
```

## Model Sizes to Expect

- **UNet**: ~860MB (320 channels)
- **HintEncoder**: ~50MB
- **Total Model**: ~1GB
- **With Optimizer State**: ~3-4GB

## Quick Deploy Script

```bash
#!/bin/bash
# deploy_model.sh - One-click model deployment

# Export trained model
python scripts/export_model.py --checkpoint outputs/rtx3090_training/final_model

# Upload to Hugging Face
python scripts/upload_to_hf.py --model_path models/exported

# Update documentation
python scripts/update_readme.py --model_url "https://huggingface.co/your-username/scribble-diffusion"

echo "🎉 Model deployed successfully!"
echo "📦 Available at: https://huggingface.co/your-username/scribble-diffusion"
```

## Usage by Others

Once uploaded, others can use your model easily:

```python
from diffusers import StableDiffusionPipeline
import torch

# Load your trained model
pipe = StableDiffusionPipeline.from_pretrained(
    "your-username/scribble-diffusion-rtx3090",
    torch_dtype=torch.float16
)

# Generate image from sketch
sketch = load_sketch("my_drawing.png")
image = pipe(prompt="a beautiful landscape", image=sketch).images[0]
```

## Best Practices

1. **Use SafeTensors format** - More secure and faster loading
2. **Include model card** - Describe training data, limitations, usage
3. **Version your models** - Use semantic versioning (v1.0, v1.1, etc.)
4. **Document performance** - Include training metrics, sample outputs
5. **Provide inference code** - Make it easy for others to use

Choose the option that best fits your use case! For open-source sharing, Hugging Face Hub is usually the best choice.
