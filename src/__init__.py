"""
Package initialization for ScribbleDiffusion.
"""

from .models.unet import SketchConditionedUNet
from .models.hint_encoder import HintEncoder
from .inference.pipeline import ScribbleDiffusionPipeline
from .data.dataset import ScribbleDataset
from .utils.attention_viz import AttentionVisualizer
from .utils.evaluation import ScribbleEvaluator

__version__ = "0.1.0"
__author__ = "ScribbleDiffusion Team"

__all__ = [
    "SketchConditionedUNet",
    "HintEncoder", 
    "ScribbleDiffusionPipeline",
    "ScribbleDataset",
    "AttentionVisualizer",
    "ScribbleEvaluator",
]
