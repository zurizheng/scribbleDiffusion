"""
Package initialization for ScribbleDiffusion.
"""

# Make imports optional to avoid circular import issues
# Individual modules can be imported directly when needed

__version__ = "0.1.0"
__author__ = "ScribbleDiffusion Team"

# Lazy imports - only import when explicitly requested
def _get_unet():
    from .models.unet import SketchConditionedUNet
    return SketchConditionedUNet

def _get_hint_encoder():
    from .models.hint_encoder import HintEncoder
    return HintEncoder

def _get_pipeline():
    from .inference.pipeline import ScribbleDiffusionPipeline
    return ScribbleDiffusionPipeline

def _get_dataset():
    from .data.dataset import ScribbleDataset
    return ScribbleDataset

def _get_attention_viz():
    from .utils.attention_viz import AttentionVisualizer
    return AttentionVisualizer

def _get_evaluator():
    from .utils.evaluation import ScribbleEvaluator
    return ScribbleEvaluator

# Make classes available when accessed
__all__ = [
    "SketchConditionedUNet",
    "HintEncoder", 
    "ScribbleDiffusionPipeline",
    "ScribbleDataset",
    "AttentionVisualizer",
    "ScribbleEvaluator",
]
