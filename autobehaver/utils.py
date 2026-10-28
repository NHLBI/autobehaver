"""Manage cached memory, report GPU memory and seed data-loader workers."""

from __future__ import annotations

import gc
import random

import numpy as np
import torch


__all__ = ['cleanup_memory', 'print_gpu_memory_summary', 'seed_worker']


def cleanup_memory() -> None:
    """Release cached CUDA/Python memory where possible."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def print_gpu_memory_summary(device_id: int = 0) -> None:
    """Print a summary of GPU memory usage for a specific device."""
    if not torch.cuda.is_available():
        print("CUDA is not available. No GPU memory to report.")
        return

    gb_divisor = 1024**3

    total_memory_gb = torch.cuda.get_device_properties(device_id).total_memory / gb_divisor
    allocated_memory_gb = torch.cuda.memory_allocated(device_id) / gb_divisor
    reserved_memory_gb = torch.cuda.memory_reserved(device_id) / gb_divisor
    free_memory_gb = total_memory_gb - reserved_memory_gb

    print("\n--- GPU Memory Summary ---")
    print(f"Device: {torch.cuda.get_device_name(device_id)}")
    print(f"  - Total Memory:    {total_memory_gb:.2f} GB")
    print(f"  - Allocated Memory:  {allocated_memory_gb:.2f} GB ({allocated_memory_gb * 1024:.0f} MB)")
    print(f"  - Reserved Memory:   {reserved_memory_gb:.2f} GB ({reserved_memory_gb * 1024:.0f} MB)")
    print(f"  - Free Memory:       {free_memory_gb:.2f} GB")
    print("--------------------------\n")


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker process."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
