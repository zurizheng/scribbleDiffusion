"""
Logging utilities for ScribbleDiffusion.
"""

import logging
import sys
from pathlib import Path


def setup_logging(accelerator):
    """Setup logging configuration."""
    
    # Configure Python logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    # Set logging level based on accelerator
    if accelerator.is_local_main_process:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Disable some noisy loggers
    logging.getLogger("diffusers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
