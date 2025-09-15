#!/usr/bin/env python3
"""
Real-time GPU memory monitor
Run this in a separate terminal while training to see live memory usage
"""

import torch
import time
import psutil
import os
import signal
import sys
from datetime import datetime

def signal_handler(sig, frame):
    print('\n👋 Memory monitoring stopped.')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def format_bytes(bytes_val):
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f}TB"

def monitor_memory(interval=2):
    """Monitor GPU and system memory usage"""
    print("🔍 Starting GPU Memory Monitor")
    print("Press Ctrl+C to stop")
    print("-" * 80)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
        
    device_count = torch.cuda.device_count()
    print(f"📱 Found {device_count} CUDA device(s)")
    
    # Get GPU names
    gpu_names = []
    for i in range(device_count):
        gpu_names.append(torch.cuda.get_device_name(i))
        print(f"   GPU {i}: {gpu_names[i]}")
    
    print("-" * 80)
    
    max_memory_seen = {}
    start_time = time.time()
    
    try:
        while True:
            current_time = datetime.now().strftime("%H:%M:%S")
            elapsed = time.time() - start_time
            
            # System memory
            system_memory = psutil.virtual_memory()
            
            print(f"\n🕐 {current_time} | Elapsed: {elapsed:.0f}s")
            print(f"💾 System RAM: {format_bytes(system_memory.used)}/{format_bytes(system_memory.total)} "
                  f"({system_memory.percent:.1f}%)")
            
            # GPU memory for each device
            for gpu_id in range(device_count):
                torch.cuda.set_device(gpu_id)
                
                allocated = torch.cuda.memory_allocated(gpu_id)
                reserved = torch.cuda.memory_reserved(gpu_id)
                max_allocated = torch.cuda.max_memory_allocated(gpu_id)
                
                # Track peak memory
                if gpu_id not in max_memory_seen:
                    max_memory_seen[gpu_id] = allocated
                else:
                    max_memory_seen[gpu_id] = max(max_memory_seen[gpu_id], allocated)
                
                # Get total GPU memory
                total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
                
                allocated_gb = allocated / 1024**3
                reserved_gb = reserved / 1024**3
                total_gb = total_memory / 1024**3
                max_gb = max_allocated / 1024**3
                peak_gb = max_memory_seen[gpu_id] / 1024**3
                
                utilization = (allocated / total_memory) * 100
                
                # Color coding based on usage
                if utilization > 90:
                    status = "🔴 CRITICAL"
                elif utilization > 75:
                    status = "🟡 HIGH"
                elif utilization > 50:
                    status = "🟢 MODERATE"
                else:
                    status = "🔵 LOW"
                
                print(f"🖥️  GPU {gpu_id} ({gpu_names[gpu_id][:20]}):")
                print(f"    {status} | {allocated_gb:.2f}GB / {total_gb:.2f}GB ({utilization:.1f}%)")
                print(f"    Reserved: {reserved_gb:.2f}GB | Max this session: {max_gb:.2f}GB | Peak: {peak_gb:.2f}GB")
                
                # Show tensor count if we can
                try:
                    import gc
                    tensor_count = 0
                    for obj in gc.get_objects():
                        if torch.is_tensor(obj) and obj.is_cuda and obj.device.index == gpu_id:
                            tensor_count += 1
                    print(f"    Active GPU tensors: {tensor_count}")
                except:
                    pass
                
                # Memory warning
                if utilization > 95:
                    print("    ⚠️  WARNING: Very high memory usage - OOM risk!")
                elif allocated_gb > 22:  # RTX 3090 has ~24GB
                    print("    ⚠️  WARNING: Approaching memory limit!")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print('\n👋 Memory monitoring stopped by user.')
    except Exception as e:
        print(f'\n❌ Error in memory monitoring: {e}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Monitor GPU memory usage")
    parser.add_argument("--interval", type=float, default=2.0, help="Update interval in seconds")
    args = parser.parse_args()
    
    monitor_memory(args.interval)