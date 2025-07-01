"""
GPU Acceleration Check for Enhanced Simulation Framework

Checks for available GPU acceleration libraries and provides fallback options.
"""

import numpy as np
import logging

def check_gpu_availability():
    """Check for GPU acceleration options"""
    gpu_info = {
        'cupy_available': False,
        'numba_cuda_available': False,
        'torch_cuda_available': False,
        'recommended_backend': 'numpy',
        'gpu_devices': []
    }
    
    # Check CuPy
    try:
        import cupy as cp
        gpu_info['cupy_available'] = True
        gpu_info['recommended_backend'] = 'cupy'
        gpu_info['gpu_devices'].append(f"CuPy GPU device count: {cp.cuda.runtime.getDeviceCount()}")
    except ImportError:
        pass
    except Exception as e:
        logging.warning(f"CuPy import error: {e}")
    
    # Check Numba CUDA
    try:
        from numba import cuda
        if cuda.is_available():
            gpu_info['numba_cuda_available'] = True
            if gpu_info['recommended_backend'] == 'numpy':
                gpu_info['recommended_backend'] = 'numba_cuda'
            gpu_info['gpu_devices'].append(f"Numba CUDA devices: {len(cuda.gpus)}")
    except ImportError:
        pass
    except Exception as e:
        logging.warning(f"Numba CUDA error: {e}")
    
    # Check PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['torch_cuda_available'] = True
            gpu_info['gpu_devices'].append(f"PyTorch CUDA devices: {torch.cuda.device_count()}")
    except ImportError:
        pass
    except Exception as e:
        logging.warning(f"PyTorch CUDA error: {e}")
    
    return gpu_info

def get_array_backend(prefer_gpu=True):
    """Get appropriate array backend (GPU if available, NumPy fallback)"""
    if not prefer_gpu:
        return np, "numpy"
    
    gpu_info = check_gpu_availability()
    
    if gpu_info['cupy_available']:
        try:
            import cupy as cp
            return cp, "cupy"
        except:
            pass
    
    # Fallback to NumPy
    return np, "numpy"

if __name__ == "__main__":
    gpu_info = check_gpu_availability()
    print("GPU Acceleration Status:")
    for key, value in gpu_info.items():
        print(f"  {key}: {value}")
    
    # Test backend
    backend, backend_name = get_array_backend()
    print(f"\nUsing backend: {backend_name}")
    
    # Quick performance test
    if backend_name != "numpy":
        test_array = backend.random.random((1000, 1000))
        import time
        start = time.time()
        result = backend.dot(test_array, test_array)
        gpu_time = time.time() - start
        print(f"GPU matrix multiply time: {gpu_time:.4f}s")
    
    # NumPy comparison
    test_array_np = np.random.random((1000, 1000))
    start = time.time()
    result_np = np.dot(test_array_np, test_array_np)
    cpu_time = time.time() - start
    print(f"CPU matrix multiply time: {cpu_time:.4f}s")
    
    if backend_name != "numpy":
        speedup = cpu_time / gpu_time
        print(f"GPU speedup: {speedup:.2f}x")
