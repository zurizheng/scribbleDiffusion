# ScribbleDiffusion

A lightweight latent diffusion model that turns rough sketches + text into full 256×256 images with live attention visualizations.

## Features

- **Sketch + Text Conditioning**: Combines edge maps with text prompts for precise control
- **ControlNet-lite Architecture**: Efficient sketch conditioning with <100M parameters
- **Dual Classifier-Free Guidance**: Separate control for text and sketch influence
- **Live Attention Visualization**: Real-time heatmaps showing word-to-region attention
- **Denoising Timeline**: Step-by-step visualization of the generation process

## Architecture

```
Input Sketch (256x256) → Edge Processing → Hint Encoder
                                              ↓
Text Prompt → CLIP Text Encoder → Cross-Attention ← U-Net (32x32x4 latents)
                                              ↓
                                          VAE Decoder → Output Image (256x256)
```

## Quick Start

```bash
# Setup environment
pip install -r requirements.txt

# Download pretrained components
python scripts/download_models.py

# Train on your dataset
python train.py --config configs/base.yaml

# Launch demo app
python app.py
```

## Project Structure

```
scribbleDiffusion/
├── src/
│   ├── models/           # Model architectures
│   ├── data/            # Dataset and preprocessing
│   ├── training/        # Training loops and losses
│   ├── inference/       # Sampling and generation
│   └── utils/           # Utilities and helpers
├── configs/             # Training configurations
├── scripts/             # Setup and utility scripts
├── notebooks/           # Experimentation notebooks
├── demo/               # Web demo application
└── tests/              # Unit tests
```

## Key Components

### Model Architecture
- **VAE**: Pretrained encoder/decoder (frozen)
- **Text Encoder**: CLIP text model (frozen) 
- **U-Net**: Custom lightweight architecture (trainable)
- **Hint Encoder**: Sketch conditioning network (trainable)

### Training Features
- Mixed precision training
- Gradient checkpointing
- EMA model weights
- Classifier-free guidance for both text and sketch
- Aggressive edge augmentations

### Inference Controls
- Text CFG scale (how much to follow text)
- Sketch CFG scale (how much to follow sketch)
- Number of denoising steps
- Spatial sketch masking
- Real-time attention visualization

## Evaluation

- **Qualitative**: Multi-prompt grids, ablation comparisons
- **Quantitative**: CLIP scores, edge fidelity metrics
- **Interactive**: A/B testing interface

## Research Contributions

1. Lightweight sketch conditioning architecture
2. Dual guidance mechanism for text + sketch
3. Live attention visualization system
4. Comprehensive ablation studies on injection methods

## License

MIT License - see LICENSE file for details.
