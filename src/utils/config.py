"""
Configuration utilities for ScribbleDiffusion.
"""

from omegaconf import OmegaConf
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> OmegaConf:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration YAML file
        
    Returns:
        OmegaConf configuration object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load the main config
    config = OmegaConf.load(config_path)
    
    # Add some default values if missing
    config = _add_default_values(config)
    
    return config


def _add_default_values(config: OmegaConf) -> OmegaConf:
    """Add default values for missing configuration fields."""
    
    # Set default training values
    if "training" not in config:
        config.training = {}
    
    training_defaults = {
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "epsilon": 1e-8,
        "optimizer": "adamw",
        "max_train_steps": 1000,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "mixed_precision": "fp16",
        "gradient_clipping": 1.0,
        "use_ema": False,
        "ema_decay": 0.9999,
        "seed": 42,
    }
    
    for key, value in training_defaults.items():
        if key not in config.training:
            config.training[key] = value
    
    # Set default data values
    if "data" not in config:
        config.data = {}
    
    data_defaults = {
        "dataset_name": "coco",
        "dataset_type": "coco",
        "image_size": 512,
        "batch_size": 1,
        "data_root": "./data",
        "limit_dataset_size": None,
        "download_coco": True,
    }
    
    for key, value in data_defaults.items():
        if key not in config.data:
            config.data[key] = value
    
    # Set default logging values
    if "logging" not in config:
        config.logging = {}
    
    logging_defaults = {
        "log_interval": 10,
        "save_interval": 500,
        "project_name": "scribble-diffusion",
        "run_name": "training",
    }
    
    for key, value in logging_defaults.items():
        if key not in config.logging:
            config.logging[key] = value
    
    # Set default validation values
    if "validation" not in config:
        config.validation = {}
    
    validation_defaults = {
        "validation_steps": 500,
        "num_validation_images": 4,
        "guidance_scale": 7.5,
        "num_inference_steps": 50,
    }
    
    for key, value in validation_defaults.items():
        if key not in config.validation:
            config.validation[key] = value
    
    return config


def save_config(config: OmegaConf, output_path: str):
    """Save configuration to YAML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        OmegaConf.save(config, f)


def merge_configs(base_config: OmegaConf, override_config: OmegaConf) -> OmegaConf:
    """Merge two configurations, with override_config taking precedence."""
    return OmegaConf.merge(base_config, override_config)


def config_to_dict(config: OmegaConf) -> Dict[str, Any]:
    """Convert OmegaConf to regular Python dictionary."""
    return OmegaConf.to_container(config, resolve=True)