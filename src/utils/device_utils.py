"""
Device utilities for ScribbleDiffusion - supports CUDA, MPS (Apple Silicon), and CPU
"""

import torch

def get_optimal_device(force_cpu=False):
    """
    Get the optimal device for PyTorch operations.
    
    Priority:
    1. CUDA (NVIDIA GPUs) if available
    2. MPS (Apple Silicon GPUs) if available  
    3. CPU as fallback
    
    Args:
        force_cpu (bool): Force CPU usage even if GPU is available
        
    Returns:
        torch.device: The optimal device
    """
    if force_cpu:
        return torch.device("cpu")
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def clear_device_cache():
    """Clear device cache for the current device."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS cache clearing (if supported in future PyTorch versions)
        if hasattr(torch.backends.mps, 'empty_cache'):
            torch.backends.mps.empty_cache()

def synchronize_device():
    """Synchronize the current device."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # MPS synchronization not needed/available

def get_device_memory_gb():
    """Get current device memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    # MPS memory tracking not available yet
    return 0

def get_device_name():
    """Get the name of the current device."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "Apple Silicon (MPS)"
    else:
        return "CPU"

def set_memory_fraction(fraction=0.95):
    """Set memory fraction for GPU devices."""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)
    # MPS memory fraction not available

def is_gpu_available():
    """Check if any GPU (CUDA or MPS) is available."""
    return (torch.cuda.is_available() or 
            (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()))

def get_device_info():
    """Get comprehensive device information."""
    device = get_optimal_device()
    
    info = {
        'device': device,
        'device_name': get_device_name(),
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'gpu_available': is_gpu_available()
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return info