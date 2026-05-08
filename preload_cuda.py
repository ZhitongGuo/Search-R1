"""Pre-load CUDA libraries with RTLD_GLOBAL to fix vllm symbol resolution."""
import ctypes
import os

cuda_lib = os.environ.get("CUDA_HOME", "/usr/local/cuda") + "/lib64"
ctypes.CDLL(f"{cuda_lib}/libcublas.so.12", mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(f"{cuda_lib}/libcublasLt.so.12", mode=ctypes.RTLD_GLOBAL)
