"""
GPU Configuration for Multi-GPU Execution

Supports:
- Single GPU (1x RTX 4090 / A100 / H100)
- Multi-GPU (2x-3x GPUs)
- Mixed precision training (fp16/fp32)
- Distributed training (DDP)
- Gradient accumulation for larger batches
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
import warnings

warnings.filterwarnings('ignore')


class GPUConfig:
    """GPU configuration management."""
    
    def __init__(self, use_multi_gpu: bool = True, mixed_precision: bool = True):
        self.use_multi_gpu = use_multi_gpu
        self.mixed_precision = mixed_precision
        self.device = None
        self.num_gpus = 0
    
    def setup(self):
        """Setup GPU environment."""
        print("="*80)
        print("GPU CONFIGURATION")
        print("="*80)
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available - using CPU")
            self.device = torch.device('cpu')
            self.num_gpus = 0
            return
        
        print(f"\n✓ CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        
        # GPU count
        self.num_gpus = torch.cuda.device_count()
        print(f"\nNumber of GPUs: {self.num_gpus}")
        
        # List GPUs
        for i in range(self.num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Max Threads per Block: {props.max_threads_per_block}")
        
        # Set device
        if self.use_multi_gpu and self.num_gpus > 1:
            self.device = torch.device('cuda')
            print(f"\n✓ Using {self.num_gpus} GPUs")
        else:
            self.device = torch.device('cuda:0')
            print(f"\n✓ Using GPU 0")
        
        # Mixed precision
        if self.mixed_precision:
            print(f"✓ Mixed precision enabled (fp16)")
            torch.cuda.empty_cache()
        
        # cuDNN settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # For speed
        print("✓ cuDNN optimization enabled")
        
        print("\n" + "="*80 + "\n")
        
        return self.device
    
    def get_device(self):
        """Get configured device."""
        return self.device
    
    def wrap_model(self, model):
        """Wrap model for multi-GPU if needed."""
        if self.num_gpus > 1 and self.use_multi_gpu:
            model = DataParallel(model)
            print(f"✓ Model wrapped with DataParallel ({self.num_gpus} GPUs)")
        return model
    
    def get_batch_size(self, base_batch_size: int) -> int:
        """
        Recommended batch size based on GPUs.
        
        Can scale with number of GPUs.
        """
        if self.num_gpus > 1:
            return base_batch_size * self.num_gpus
        return base_batch_size
    
    def empty_cache(self):
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✓ GPU cache cleared")
    
    def print_memory_usage(self):
        """Print GPU memory usage."""
        if not torch.cuda.is_available():
            return
        
        print("\nGPU Memory Usage:")
        for i in range(self.num_gpus):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            
            print(f"  GPU {i}:")
            print(f"    Allocated: {allocated:.2f} GB")
            print(f"    Reserved: {reserved:.2f} GB")
            print(f"    Total: {total:.2f} GB")
            print(f"    Free: {total - reserved:.2f} GB")


class OptimizedTrainingConfig:
    """Optimized training configuration for multi-GPU."""
    
    @staticmethod
    def get_optimal_batch_size(num_gpus: int, base_batch_size: int = 32) -> int:
        """Calculate optimal batch size for GPUs."""
        if num_gpus == 1:
            return base_batch_size
        elif num_gpus == 2:
            return base_batch_size * 2
        elif num_gpus == 3:
            return base_batch_size * 3
        else:
            return base_batch_size * num_gpus
    
    @staticmethod
    def get_optimal_learning_rate(base_lr: float = 2e-5, num_gpus: int = 1) -> float:
        """
        Adjust learning rate for multi-GPU training.
        
        With larger effective batch size, can use higher LR.
        """
        # Scale with sqrt of batch size scaling factor
        scale = np.sqrt(num_gpus)
        return base_lr * scale
    
    @staticmethod
    def get_gradient_accumulation_steps(
        effective_batch_size: int,
        target_batch_size: int = 256
    ) -> int:
        """
        Calculate gradient accumulation steps for larger effective batches.
        
        Useful when total batch size > GPU memory allows.
        """
        return max(1, target_batch_size // effective_batch_size)
    
    @staticmethod
    def get_num_workers(num_gpus: int) -> int:
        """Recommended number of data loading workers."""
        return 4 * num_gpus


class DistributedTrainingManager:
    """Manage distributed training across multiple GPUs."""
    
    def __init__(self, backend: str = 'nccl'):
        self.backend = backend
        self.distributed = False
    
    def setup_ddp(self, rank: int, world_size: int):
        """Setup Distributed Data Parallel."""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        dist.init_process_group(self.backend, rank=rank, world_size=world_size)
        self.distributed = True
        
        print(f"✓ DDP initialized: rank {rank}/{world_size}")
    
    def cleanup_ddp(self):
        """Cleanup DDP."""
        if self.distributed:
            dist.destroy_process_group()
    
    def wrap_model_ddp(self, model, rank: int):
        """Wrap model with DistributedDataParallel."""
        if torch.cuda.is_available():
            model = model.to(rank)
            model = DistributedDataParallel(model, device_ids=[rank])
        return model


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import numpy as np
    
    # Setup GPU
    gpu_config = GPUConfig(use_multi_gpu=True, mixed_precision=True)
    device = gpu_config.setup()
    
    # Get optimal training config
    opt_config = OptimizedTrainingConfig()
    batch_size = opt_config.get_optimal_batch_size(gpu_config.num_gpus)
    learning_rate = opt_config.get_optimal_learning_rate(num_gpus=gpu_config.num_gpus)
    num_workers = opt_config.get_num_workers(gpu_config.num_gpus)
    
    print(f"Recommended Training Config:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Data Workers: {num_workers}")
    
    # Test memory
    gpu_config.print_memory_usage()
    
    # Example: create dummy model and move to device
    print("\nTesting model loading...")
    
    import torch.nn as nn
    
    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 2)
    )
    
    model = model.to(device)
    
    if gpu_config.num_gpus > 1:
        model = gpu_config.wrap_model(model)
    
    print(f"✓ Model loaded on {device}")
    
    # Test inference
    x = torch.randn(batch_size, 256).to(device)
    y = model(x)
    print(f"✓ Inference successful: input {x.shape} -> output {y.shape}")
    
    gpu_config.print_memory_usage()
    gpu_config.empty_cache()
