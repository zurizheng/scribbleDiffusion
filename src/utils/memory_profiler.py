"""
GPU Memory Profiler for ScribbleDiffusion Training
Tracks memory usage at each step to identify memory bottlenecks
"""

import torch
import gc
import psutil
import os
from typing import Dict, List, Tuple
import time

class GPUMemoryProfiler:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self.profile_data = []
        self.step_counter = 0
        
        if self.enabled:
            # Enable memory fraction tracking
            torch.cuda.memory._record_memory_history()
            
    def profile_memory(self, step_name: str, details: str = "") -> Dict:
        """Profile current GPU memory usage"""
        if not self.enabled:
            return {}
            
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            # Get memory summary
            memory_summary = torch.cuda.memory_summary()
            
            profile_info = {
                'step': self.step_counter,
                'step_name': step_name,
                'details': details,
                'allocated_gb': allocated,
                'reserved_gb': reserved, 
                'max_allocated_gb': max_allocated,
                'free_gb': reserved - allocated,
                'timestamp': time.time()
            }
            
            self.profile_data.append(profile_info)
            
            # Print immediate feedback
            print(f"🔍 {step_name}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {details}")
            
            # Check for concerning memory usage
            if allocated > 20.0:  # > 20GB on RTX 3090
                print(f"⚠️  HIGH MEMORY: {allocated:.2f}GB allocated!")
                self.print_memory_breakdown()
                
            return profile_info
            
        except Exception as e:
            print(f"Memory profiling error: {e}")
            return {}
    
    def print_memory_breakdown(self):
        """Print detailed memory breakdown"""
        if not self.enabled:
            return
            
        print("\n" + "="*60)
        print("DETAILED GPU MEMORY BREAKDOWN")
        print("="*60)
        
        try:
            # Get memory summary
            summary = torch.cuda.memory_summary()
            print(summary)
            
            # Show tensor count and sizes
            tensors = []
            for obj in gc.get_objects():
                if torch.is_tensor(obj) and obj.is_cuda:
                    tensors.append((obj.dtype, obj.shape, obj.numel() * obj.element_size() / 1024**2))  # MB
            
            # Sort by size
            tensors.sort(key=lambda x: x[2], reverse=True)
            
            print(f"\nTop 10 GPU Tensors:")
            for i, (dtype, shape, size_mb) in enumerate(tensors[:10]):
                print(f"  {i+1}. {dtype} {shape} - {size_mb:.1f}MB")
                
            total_tensor_mb = sum(t[2] for t in tensors)
            print(f"\nTotal tensor memory: {total_tensor_mb:.1f}MB ({total_tensor_mb/1024:.2f}GB)")
            print(f"Number of GPU tensors: {len(tensors)}")
            
        except Exception as e:
            print(f"Error in memory breakdown: {e}")
        
        print("="*60 + "\n")
    
    def checkpoint_memory(self, name: str):
        """Save memory checkpoint for comparison"""
        self.profile_memory(f"CHECKPOINT_{name}")
    
    def step(self):
        """Increment step counter"""
        self.step_counter += 1
    
    def save_profile(self, filepath: str):
        """Save profiling data to file"""
        if not self.profile_data:
            return
            
        import json
        with open(filepath, 'w') as f:
            json.dump(self.profile_data, f, indent=2)
        print(f"Memory profile saved to {filepath}")
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage in GB"""
        if not self.profile_data:
            return 0.0
        return max(p['allocated_gb'] for p in self.profile_data)


def profile_model_memory(model, model_name: str, profiler: GPUMemoryProfiler):
    """Profile memory usage of a specific model"""
    if not torch.cuda.is_available():
        return
        
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory (roughly)
    param_memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    
    details = f"{total_params:,} params ({trainable_params:,} trainable), ~{param_memory_mb:.1f}MB"
    profiler.profile_memory(f"MODEL_{model_name}", details)


def profile_tensor_operation(func, name: str, profiler: GPUMemoryProfiler, *args, **kwargs):
    """Profile memory usage of a tensor operation"""
    profiler.profile_memory(f"BEFORE_{name}")
    
    try:
        result = func(*args, **kwargs)
        profiler.profile_memory(f"AFTER_{name}")
        return result
    except torch.cuda.OutOfMemoryError as e:
        profiler.profile_memory(f"OOM_{name}")
        profiler.print_memory_breakdown()
        raise e


# Context manager for automatic profiling
class MemoryProfileContext:
    def __init__(self, profiler: GPUMemoryProfiler, name: str):
        self.profiler = profiler
        self.name = name
        
    def __enter__(self):
        self.profiler.profile_memory(f"START_{self.name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is torch.cuda.OutOfMemoryError:
            self.profiler.profile_memory(f"OOM_{self.name}")
            self.profiler.print_memory_breakdown()
        else:
            self.profiler.profile_memory(f"END_{self.name}")


# Example usage functions
def profile_forward_pass(model, inputs, profiler: GPUMemoryProfiler):
    """Profile a forward pass"""
    with MemoryProfileContext(profiler, "FORWARD_PASS"):
        return model(**inputs)


def profile_backward_pass(loss, profiler: GPUMemoryProfiler):
    """Profile a backward pass"""
    with MemoryProfileContext(profiler, "BACKWARD_PASS"):
        loss.backward()


def emergency_memory_cleanup():
    """Emergency memory cleanup when OOM occurs"""
    print("🚨 EMERGENCY MEMORY CLEANUP")
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection
    gc.collect()
    
    # Show current memory state
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"After cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")