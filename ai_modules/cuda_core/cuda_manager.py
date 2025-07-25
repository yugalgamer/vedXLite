"""
CUDA Core Manager
================
Comprehensive GPU acceleration for all AI components in the system.
Manages CUDA resources, memory, and device allocation across all modules.
"""

import torch
import torch.nn.functional as F
from torch import nn
import logging
import time
import gc
import psutil
from typing import Dict, Any, List, Optional, Tuple, Union
import threading
from dataclasses import dataclass
from contextlib import contextmanager
import json

logger = logging.getLogger(__name__)

@dataclass
class CudaDeviceInfo:
    """Information about a CUDA device."""
    index: int
    name: str
    total_memory: float  # GB
    compute_capability: Tuple[int, int]
    multiprocessor_count: int
    is_available: bool = True

@dataclass
class CudaMemoryInfo:
    """Current CUDA memory usage information."""
    allocated: float  # GB
    reserved: float   # GB
    max_allocated: float  # GB
    total: float     # GB
    free: float      # GB
    utilization: float  # Percentage

class CudaManager:
    """
    Comprehensive CUDA manager for all AI operations.
    Handles device allocation, memory management, and performance optimization.
    """
    
    def __init__(self, auto_optimize: bool = True):
        """
        Initialize the CUDA manager.
        
        Args:
            auto_optimize: Whether to automatically optimize CUDA settings
        """
        self.auto_optimize = auto_optimize
        self.device_lock = threading.Lock()
        self.memory_threshold = 0.85  # Use up to 85% of GPU memory
        
        # Initialize CUDA
        self.cuda_available = torch.cuda.is_available()
        self.devices = []
        self.current_device = None
        self.default_device = None
        
        # Performance tracking
        self.stats = {
            'total_operations': 0,
            'cuda_operations': 0,
            'cpu_operations': 0,
            'memory_errors': 0,
            'device_switches': 0,
            'average_processing_time': 0.0,
            'total_memory_allocated': 0.0
        }
        
        self._initialize_cuda()
        
    def _initialize_cuda(self):
        """Initialize CUDA devices and settings."""
        if not self.cuda_available:
            logger.warning("⚠️  CUDA not available - using CPU only")
            return
        
        try:
            # Get device information
            device_count = torch.cuda.device_count()
            logger.info(f"🚀 Found {device_count} CUDA device(s)")
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                device_info = CudaDeviceInfo(
                    index=i,
                    name=props.name,
                    total_memory=props.total_memory / (1024**3),
                    compute_capability=(props.major, props.minor),
                    multiprocessor_count=props.multi_processor_count
                )
                self.devices.append(device_info)
                
                logger.info(f"   📱 Device {i}: {device_info.name}")
                logger.info(f"      Memory: {device_info.total_memory:.1f} GB")
                logger.info(f"      Compute: {device_info.compute_capability}")
            
            # Set default device (usually the first one)
            self.current_device = torch.device(f"cuda:{self.devices[0].index}")
            self.default_device = self.current_device
            torch.cuda.set_device(self.current_device)
            
            # Optimize CUDA settings
            if self.auto_optimize:
                self._optimize_cuda_settings()
                
            logger.info(f"✅ CUDA Manager initialized on {self.current_device}")
            
        except Exception as e:
            logger.error(f"❌ CUDA initialization failed: {e}")
            self.cuda_available = False
    
    def _optimize_cuda_settings(self):
        """Optimize CUDA settings for best performance."""
        try:
            # Enable optimizations
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
            
            # Set memory management
            torch.cuda.empty_cache()
            
            # Enable mixed precision if supported
            if self._supports_mixed_precision():
                logger.info("✅ Mixed precision (FP16) support enabled")
            
            logger.info("⚡ CUDA optimizations applied")
            
        except Exception as e:
            logger.warning(f"⚠️  CUDA optimization failed: {e}")
    
    def _supports_mixed_precision(self) -> bool:
        """Check if the device supports mixed precision (FP16)."""
        if not self.cuda_available or not self.devices:
            return False
        
        # Check compute capability (7.0+ supports Tensor Cores)
        device = self.devices[0]
        return device.compute_capability[0] >= 7
    
    @contextmanager
    def cuda_context(self, device: Optional[Union[str, torch.device]] = None):
        """
        Context manager for CUDA operations.
        
        Args:
            device: Device to use for this context
        """
        original_device = self.current_device
        
        try:
            if device is not None:
                self.set_device(device)
            
            yield self.current_device
            
        finally:
            if device is not None and original_device != self.current_device:
                self.set_device(original_device)
    
    def set_device(self, device: Union[str, int, torch.device]):
        """
        Set the active CUDA device.
        
        Args:
            device: Device to set as active
        """
        with self.device_lock:
            try:
                if isinstance(device, str):
                    new_device = torch.device(device)
                elif isinstance(device, int):
                    new_device = torch.device(f"cuda:{device}")
                else:
                    new_device = device
                
                if new_device != self.current_device:
                    torch.cuda.set_device(new_device)
                    self.current_device = new_device
                    self.stats['device_switches'] += 1
                    logger.debug(f"🔄 Switched to device: {new_device}")
                
            except Exception as e:
                logger.error(f"❌ Failed to set device {device}: {e}")
    
    def get_optimal_device(self, memory_required: float = 0.0) -> torch.device:
        """
        Get the optimal device for a task based on memory requirements.
        
        Args:
            memory_required: Memory required in GB
            
        Returns:
            Optimal device to use
        """
        if not self.cuda_available:
            return torch.device("cpu")
        
        # For now, return the default device
        # In future, could implement load balancing across multiple GPUs
        return self.default_device
    
    def get_memory_info(self, device: Optional[torch.device] = None) -> CudaMemoryInfo:
        """
        Get detailed memory information for a device.
        
        Args:
            device: Device to check (uses current device if None)
            
        Returns:
            Memory information
        """
        if not self.cuda_available:
            return CudaMemoryInfo(0, 0, 0, 0, 0, 0)
        
        device = device or self.current_device
        
        try:
            with torch.cuda.device(device):
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
                
                # Get total memory from device properties
                device_idx = device.index if device.type == 'cuda' else 0
                total = self.devices[device_idx].total_memory
                free = total - allocated
                utilization = (allocated / total) * 100 if total > 0 else 0
                
                return CudaMemoryInfo(
                    allocated=allocated,
                    reserved=reserved,
                    max_allocated=max_allocated,
                    total=total,
                    free=free,
                    utilization=utilization
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to get memory info: {e}")
            return CudaMemoryInfo(0, 0, 0, 0, 0, 0)
    
    def check_memory_available(self, required_gb: float, device: Optional[torch.device] = None) -> bool:
        """
        Check if enough memory is available for an operation.
        
        Args:
            required_gb: Required memory in GB
            device: Device to check
            
        Returns:
            True if enough memory is available
        """
        if not self.cuda_available:
            return True  # CPU operations
        
        memory_info = self.get_memory_info(device)
        available = memory_info.free
        threshold = memory_info.total * self.memory_threshold
        
        return (memory_info.allocated + required_gb) <= threshold
    
    def clear_cache(self, device: Optional[torch.device] = None):
        """
        Clear GPU cache to free memory.
        
        Args:
            device: Device to clear cache for
        """
        if not self.cuda_available:
            return
        
        try:
            if device:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            logger.debug("🔄 GPU cache cleared")
            
        except Exception as e:
            logger.error(f"❌ Failed to clear cache: {e}")
    
    def optimize_model_for_cuda(self, model: nn.Module, use_half_precision: bool = None) -> nn.Module:
        """
        Optimize a PyTorch model for CUDA.
        
        Args:
            model: Model to optimize
            use_half_precision: Whether to use FP16 (auto-detect if None)
            
        Returns:
            Optimized model
        """
        if not self.cuda_available:
            return model
        
        try:
            # Move to GPU
            model = model.to(self.current_device)
            
            # Set to eval mode for inference
            model.eval()
            
            # Use half precision if supported and requested
            if use_half_precision is None:
                use_half_precision = self._supports_mixed_precision()
            
            if use_half_precision:
                model = model.half()
                logger.debug("⚡ Model converted to half precision (FP16)")
            
            # Compile model if available (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model)
                    logger.debug("⚡ Model compiled with torch.compile")
                except Exception as e:
                    logger.debug(f"⚠️  Model compilation failed: {e}")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Model optimization failed: {e}")
            return model
    
    def create_tensor(self, data, dtype=None, device=None) -> torch.Tensor:
        """
        Create a tensor on the optimal device.
        
        Args:
            data: Data for tensor
            dtype: Data type
            device: Target device
            
        Returns:
            Tensor on the specified device
        """
        target_device = device or self.get_optimal_device()
        
        if isinstance(data, torch.Tensor):
            tensor = data.to(target_device)
        else:
            tensor = torch.tensor(data, device=target_device)
        
        if dtype:
            tensor = tensor.to(dtype)
        
        return tensor
    
    def profile_operation(self, operation_name: str = "operation"):
        """
        Decorator to profile CUDA operations.
        
        Args:
            operation_name: Name of the operation for logging
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                device_used = "cpu"
                
                try:
                    if self.cuda_available and self.current_device.type == 'cuda':
                        device_used = str(self.current_device)
                        torch.cuda.synchronize()  # Ensure GPU operations are complete
                    
                    result = func(*args, **kwargs)
                    
                    if self.cuda_available and self.current_device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    processing_time = time.time() - start_time
                    
                    # Update stats
                    self.stats['total_operations'] += 1
                    if device_used.startswith('cuda'):
                        self.stats['cuda_operations'] += 1
                    else:
                        self.stats['cpu_operations'] += 1
                    
                    # Update average processing time
                    total_ops = self.stats['total_operations']
                    current_avg = self.stats['average_processing_time']
                    self.stats['average_processing_time'] = (
                        (current_avg * (total_ops - 1) + processing_time) / total_ops
                    )
                    
                    logger.debug(f"⚡ {operation_name}: {processing_time:.3f}s on {device_used}")
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"❌ {operation_name} failed: {e}")
                    raise
                    
            return wrapper
        return decorator
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        info = {
            'cuda_available': self.cuda_available,
            'torch_version': torch.__version__,
            'devices': [],
            'current_device': str(self.current_device) if self.current_device else None,
            'stats': self.stats.copy()
        }
        
        if self.cuda_available:
            info['cuda_version'] = torch.version.cuda
            info['cudnn_version'] = torch.backends.cudnn.version()
            info['cudnn_enabled'] = torch.backends.cudnn.enabled
            
            for device in self.devices:
                device_info = {
                    'index': device.index,
                    'name': device.name,
                    'total_memory_gb': device.total_memory,
                    'compute_capability': device.compute_capability,
                    'multiprocessor_count': device.multiprocessor_count,
                    'memory_info': self.get_memory_info(torch.device(f"cuda:{device.index}")).__dict__
                }
                info['devices'].append(device_info)
        
        # Add CPU info
        info['cpu_info'] = {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'cpu_percent': psutil.cpu_percent()
        }
        
        return info
    
    def benchmark_device(self, device: Optional[torch.device] = None, 
                        operations: int = 1000) -> Dict[str, float]:
        """
        Benchmark a device's performance.
        
        Args:
            device: Device to benchmark
            operations: Number of operations to perform
            
        Returns:
            Benchmark results
        """
        device = device or self.current_device
        
        results = {
            'device': str(device),
            'matrix_mult_time': 0.0,
            'memory_bandwidth': 0.0,
            'tensor_ops_per_second': 0.0
        }
        
        try:
            with torch.cuda.device(device) if device.type == 'cuda' else torch.no_grad():
                # Matrix multiplication benchmark
                size = 1024
                a = torch.randn(size, size, device=device)
                b = torch.randn(size, size, device=device)
                
                start_time = time.time()
                for _ in range(operations // 100):  # Fewer ops for matrix mult
                    c = torch.mm(a, b)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                results['matrix_mult_time'] = time.time() - start_time
                
                # Tensor operations benchmark
                x = torch.randn(10000, device=device)
                
                start_time = time.time()
                for _ in range(operations):
                    y = torch.sin(x) + torch.cos(x)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                tensor_time = time.time() - start_time
                results['tensor_ops_per_second'] = operations / tensor_time
                
                logger.info(f"📊 Benchmark completed for {device}")
                
        except Exception as e:
            logger.error(f"❌ Benchmark failed for {device}: {e}")
        
        return results
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            'total_operations': 0,
            'cuda_operations': 0,
            'cpu_operations': 0,
            'memory_errors': 0,
            'device_switches': 0,
            'average_processing_time': 0.0,
            'total_memory_allocated': 0.0
        }
        
        if self.cuda_available:
            torch.cuda.reset_peak_memory_stats()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check."""
        health = {
            'overall_status': 'healthy',
            'cuda_status': 'ok' if self.cuda_available else 'unavailable',
            'memory_status': 'ok',
            'performance_status': 'ok',
            'issues': []
        }
        
        if self.cuda_available:
            # Check memory usage
            memory_info = self.get_memory_info()
            if memory_info.utilization > 90:
                health['memory_status'] = 'warning'
                health['issues'].append(f"High GPU memory usage: {memory_info.utilization:.1f}%")
            
            # Check for memory errors
            if self.stats['memory_errors'] > 0:
                health['memory_status'] = 'error'
                health['issues'].append(f"Memory errors detected: {self.stats['memory_errors']}")
            
            # Check performance
            if self.stats['total_operations'] > 0:
                cuda_ratio = self.stats['cuda_operations'] / self.stats['total_operations']
                if cuda_ratio < 0.5:
                    health['performance_status'] = 'warning'
                    health['issues'].append(f"Low GPU utilization: {cuda_ratio:.2%}")
        
        if health['issues']:
            health['overall_status'] = 'warning' if health['memory_status'] != 'error' else 'error'
        
        return health

# Global CUDA manager instance
_cuda_manager = None

def get_cuda_manager(auto_optimize: bool = True) -> CudaManager:
    """Get or create the global CUDA manager."""
    global _cuda_manager
    if _cuda_manager is None:
        _cuda_manager = CudaManager(auto_optimize=auto_optimize)
    return _cuda_manager

def cuda_available() -> bool:
    """Check if CUDA is available."""
    return get_cuda_manager().cuda_available

def get_optimal_device() -> torch.device:
    """Get the optimal device for operations."""
    return get_cuda_manager().get_optimal_device()

def clear_cuda_cache():
    """Clear CUDA cache."""
    get_cuda_manager().clear_cache()

# Test the CUDA manager if run directly
if __name__ == "__main__":
    print("🧪 Testing CUDA Manager...")
    
    manager = CudaManager()
    system_info = manager.get_system_info()
    
    print(f"📊 System Info:")
    print(f"   CUDA Available: {system_info['cuda_available']}")
    print(f"   PyTorch: {system_info['torch_version']}")
    
    if system_info['cuda_available']:
        print(f"   CUDA: {system_info['cuda_version']}")
        print(f"   Devices: {len(system_info['devices'])}")
        for device in system_info['devices']:
            print(f"      {device['name']}: {device['total_memory_gb']:.1f} GB")
    
    # Run benchmark
    if manager.cuda_available:
        print("\n🏃 Running benchmark...")
        benchmark = manager.benchmark_device()
        print(f"   Matrix multiplication: {benchmark['matrix_mult_time']:.3f}s")
        print(f"   Tensor ops/sec: {benchmark['tensor_ops_per_second']:.0f}")
    
    # Health check
    health = manager.health_check()
    print(f"\n🏥 Health: {health['overall_status']}")
    if health['issues']:
        for issue in health['issues']:
            print(f"   ⚠️  {issue}")
    
    print("✅ CUDA Manager test complete!")
