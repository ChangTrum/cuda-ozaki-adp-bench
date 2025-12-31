# -*- coding: utf-8 -*-
"""
Single-file CUDA 13.x Tensor-Network benchmark suite for Blackwell / CC 10.x-12.x
with Ozaki ADP (cuBLAS fixed-point FP64 emulation) via auto-built NVCC DLL + ctypes.

Benchmarks:
- Tensor Core GEMM (FP16/BF16/TF32/FP64).
- Ozaki ADP FP64 emulation GEMM via cuBLAS fixed-point (DLL).
- TEBD-style two-site MPS gate contraction.
- MPO/MPS environment contraction.
- Truncated SVD workload.
- Device-to-device bandwidth.

Notes on Ozaki ADP validation:
- Reports abs_err_max, relFro(err)=||E||_F/||Ref||_F, and a scaled rel_err_max to avoid
  near-zero reference entries dominating max relative error.

Usage (example):
  python cuda_hpc_bench_single.py --backend cupy --matrix-size 4096 --repeat 10 --bf16 --validate-ozaki cpu
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Backend selection
# =============================================================================


@dataclass
class Backend:
    name: str
    lib: Any
    is_torch: bool

    def synchronize(self) -> None:
        if self.is_torch:
            self.lib.cuda.synchronize()
        else:
            self.lib.cuda.Stream.null.synchronize()

    @property
    def float16(self):
        return self.lib.float16

    @property
    def bfloat16(self):
        return getattr(self.lib, "bfloat16", None)

    @property
    def float32(self):
        return self.lib.float32

    @property
    def float64(self):
        return self.lib.float64


def _import_cupy() -> Optional[Backend]:
    try:
        import cupy as cp  # type: ignore
    except Exception:
        return None

    # Try enabling Tensor Core math in cuBLAS handle (best-effort).
    try:
        handle = cp.cuda.get_cublas_handle()
        cp.cuda.cublas.setMathMode(handle, cp.cuda.cublas.CUBLAS_TENSOR_OP_MATH)
    except Exception:
        pass

    return Backend(name="cupy", lib=cp, is_torch=False)


def _import_torch() -> Optional[Backend]:
    try:
        import torch  # type: ignore
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    return Backend(name="torch", lib=torch, is_torch=True)


def choose_backend(preferred: str) -> Backend:
    preferred = preferred.lower()
    backends: List[Backend] = []

    if preferred in ("auto", "cupy"):
        b = _import_cupy()
        if b:
            backends.append(b)
        if preferred == "cupy" and not backends:
            raise RuntimeError(
                "CuPy not available; install cupy-cuda13x for CUDA 13.1."
            )
    if preferred in ("auto", "torch"):
        b = _import_torch()
        if b:
            backends.append(b)
        if preferred == "torch" and not backends:
            raise RuntimeError("PyTorch with CUDA support is required but not found.")

    if not backends:
        raise RuntimeError(
            "No GPU backend found. Install CuPy (preferred) or PyTorch with CUDA 13.1."
        )
    return backends[0]


# =============================================================================
# Utility helpers
# =============================================================================


def device_info(backend: Backend) -> Dict[str, Any]:
    if backend.is_torch:
        torch = backend.lib
        dev = torch.cuda.get_device_properties(torch.cuda.current_device())
        return {
            "name": dev.name,
            "cc_major": int(dev.major),
            "cc_minor": int(dev.minor),
            "cc": f"{dev.major}.{dev.minor}",
            "sm_count": getattr(dev, "multi_processor_count", None),
            "total_mem_gb": dev.total_memory / 1e9,
        }
    cp = backend.lib
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = (
        props["name"].decode()
        if isinstance(props["name"], (bytes, bytearray))
        else props["name"]
    )
    return {
        "name": name,
        "cc_major": int(props["major"]),
        "cc_minor": int(props["minor"]),
        "cc": f"{props['major']}.{props['minor']}",
        "sm_count": props.get("multiProcessorCount"),
        "total_mem_gb": props["totalGlobalMem"] / 1e9,
    }


def format_row(name: str, metric: float, unit: str, notes: str) -> str:
    return f"{name:<28} {metric:>10.2f} {unit:<6}  {notes}"


def measure_gpu_seconds(
    backend: Backend, fn: Callable[[], Any], repeat: int, warmup: int = 3
) -> float:
    """Time a GPU callable using events to avoid CPU scheduling noise."""
    if backend.is_torch:
        torch = backend.lib
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        for _ in range(warmup):
            fn()
        backend.synchronize()
        start.record()
        for _ in range(repeat):
            fn()
        stop.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(stop)
        return elapsed_ms / 1000.0 / repeat

    cp = backend.lib
    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    for _ in range(warmup):
        fn()
    backend.synchronize()
    start.record()
    for _ in range(repeat):
        fn()
    stop.record()
    stop.synchronize()
    elapsed_ms = cp.cuda.get_elapsed_time(start, stop)
    return elapsed_ms / 1000.0 / repeat


def randn(backend: Backend, shape: Tuple[int, ...], dtype: Any):
    if backend.is_torch:
        torch = backend.lib
        return torch.randn(*shape, device="cuda", dtype=dtype)
    cp = backend.lib
    if dtype not in (cp.float32, cp.float64):
        return cp.random.standard_normal(shape, dtype=cp.float32).astype(dtype)
    return cp.random.standard_normal(shape, dtype=dtype)


def zeros(backend: Backend, shape: Tuple[int, ...], dtype: Any):
    if backend.is_torch:
        torch = backend.lib
        return torch.zeros(*shape, device="cuda", dtype=dtype)
    cp = backend.lib
    return cp.zeros(shape, dtype=dtype)


def einsum(backend: Backend, expr: str, *operands):
    if backend.is_torch:
        torch = backend.lib
        return torch.einsum(expr, *operands)
    cp = backend.lib
    return cp.einsum(expr, *operands)


def matmul(backend: Backend, a, b, out=None):
    return backend.lib.matmul(a, b, out=out)


def svd(backend: Backend, x):
    if backend.is_torch:
        torch = backend.lib
        return torch.linalg.svd(x, full_matrices=False)
    cp = backend.lib
    return cp.linalg.svd(x, full_matrices=False)


def _get_cuda_stream_ptr(backend: Backend) -> int:
    """Return current CUDA stream pointer as integer."""
    if backend.is_torch:
        torch = backend.lib
        # torch.cuda.current_stream().cuda_stream is an integer handle
        return int(torch.cuda.current_stream().cuda_stream)
    cp = backend.lib
    return int(cp.cuda.get_current_stream().ptr)


# =============================================================================
# Ozaki ADP DLL (nvcc build + ctypes)
# =============================================================================

_CUDA_DDL_SOURCE = r"""
#include <cstdio>
#include <cstdint>
#include <cstring>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#if defined(_WIN32)
  #define OZAPI __declspec(dllexport)
#else
  #define OZAPI __attribute__((visibility("default")))
#endif

static void set_err(char* buf, int buflen, const char* msg) {
  if (!buf || buflen <= 0) return;
  std::snprintf(buf, (size_t)buflen, "%s", msg ? msg : "");
}

static void set_errf(char* buf, int buflen, const char* prefix, const char* detail) {
  if (!buf || buflen <= 0) return;
  std::snprintf(buf, (size_t)buflen, "%s%s%s", prefix ? prefix : "", (detail ? ": " : ""), (detail ? detail : ""));
}

static const char* cublas_status_str(cublasStatus_t s) {
  switch (s) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    default: return "CUBLAS_STATUS_<unknown>";
  }
}

struct OzakiCtx {
  int device = 0;
  cublasHandle_t handle = nullptr;
  void* workspace = nullptr;
  size_t workspace_bytes = 0;
  int* d_mantissa_bits = nullptr;
};

extern "C" {

OZAPI int ozaki_adp_create(
    int device,
    size_t workspace_bytes,
    int emulation_strategy,   // 0=DEFAULT, 1=PERFORMANT, 2=EAGER (per cuBLAS enum order)
    char* err, int errlen,
    void** out_ctx)
{
  if (!out_ctx) { set_err(err, errlen, "out_ctx is null"); return -1; }
  *out_ctx = nullptr;

  cudaError_t ce = cudaSetDevice(device);
  if (ce != cudaSuccess) { set_errf(err, errlen, "cudaSetDevice failed", cudaGetErrorString(ce)); return -2; }

  OzakiCtx* ctx = new OzakiCtx();
  ctx->device = device;
  ctx->workspace_bytes = workspace_bytes;

  cublasStatus_t st = cublasCreate(&ctx->handle);
  if (st != CUBLAS_STATUS_SUCCESS) {
    set_errf(err, errlen, "cublasCreate failed", cublas_status_str(st));
    delete ctx;
    return -3;
  }

  // Allocate workspace
  if (workspace_bytes > 0) {
    ce = cudaMalloc(&ctx->workspace, workspace_bytes);
    if (ce != cudaSuccess) {
      set_errf(err, errlen, "cudaMalloc(workspace) failed", cudaGetErrorString(ce));
      cublasDestroy(ctx->handle);
      delete ctx;
      return -4;
    }
  }

  // Allocate device int for ADP mantissa bitcount writeback
  ce = cudaMalloc(&ctx->d_mantissa_bits, sizeof(int));
  if (ce != cudaSuccess) {
    set_errf(err, errlen, "cudaMalloc(d_mantissa_bits) failed", cudaGetErrorString(ce));
    if (ctx->workspace) cudaFree(ctx->workspace);
    cublasDestroy(ctx->handle);
    delete ctx;
    return -5;
  }
  int minus_one = -1;
  ce = cudaMemcpy(ctx->d_mantissa_bits, &minus_one, sizeof(int), cudaMemcpyHostToDevice);
  if (ce != cudaSuccess) {
    set_errf(err, errlen, "cudaMemcpy(d_mantissa_bits init) failed", cudaGetErrorString(ce));
    cudaFree(ctx->d_mantissa_bits);
    if (ctx->workspace) cudaFree(ctx->workspace);
    cublasDestroy(ctx->handle);
    delete ctx;
    return -6;
  }

  // Pointer mode for alpha/beta
  st = cublasSetPointerMode(ctx->handle, CUBLAS_POINTER_MODE_HOST);
  if (st != CUBLAS_STATUS_SUCCESS) {
    set_errf(err, errlen, "cublasSetPointerMode failed", cublas_status_str(st));
    cudaFree(ctx->d_mantissa_bits);
    if (ctx->workspace) cudaFree(ctx->workspace);
    cublasDestroy(ctx->handle);
    delete ctx;
    return -7;
  }

  // Default math mode is fine; fixed-point is driven by computeType.
  st = cublasSetMathMode(ctx->handle, CUBLAS_DEFAULT_MATH);
  (void)st; // ignore if unavailable

  // Workspace for fixed-point emulation
  if (ctx->workspace && ctx->workspace_bytes > 0) {
    st = cublasSetWorkspace(ctx->handle, ctx->workspace, ctx->workspace_bytes);
    if (st != CUBLAS_STATUS_SUCCESS) {
      set_errf(err, errlen, "cublasSetWorkspace failed", cublas_status_str(st));
      cudaFree(ctx->d_mantissa_bits);
      cudaFree(ctx->workspace);
      cublasDestroy(ctx->handle);
      delete ctx;
      return -8;
    }
  }

  // Emulation strategy (DEFAULT/PERFORMANT/EAGER)
  st = cublasSetEmulationStrategy(ctx->handle, (cublasEmulationStrategy_t)emulation_strategy);
  if (st != CUBLAS_STATUS_SUCCESS) {
    set_errf(err, errlen, "cublasSetEmulationStrategy failed", cublas_status_str(st));
    cudaFree(ctx->d_mantissa_bits);
    if (ctx->workspace) cudaFree(ctx->workspace);
    cublasDestroy(ctx->handle);
    delete ctx;
    return -9;
  }

  // Fixed-point emulation controls: ADP (dynamic mantissa), max bits = 79, offset = 0,
  // and mantissaBitCountPointer must be DEVICE pointer (host pointer can trigger internal error).
  st = cublasSetFixedPointEmulationMantissaControl(ctx->handle, CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC);
  if (st != CUBLAS_STATUS_SUCCESS) {
    set_errf(err, errlen, "cublasSetFixedPointEmulationMantissaControl failed", cublas_status_str(st));
    cudaFree(ctx->d_mantissa_bits);
    if (ctx->workspace) cudaFree(ctx->workspace);
    cublasDestroy(ctx->handle);
    delete ctx;
    return -10;
  }

  st = cublasSetFixedPointEmulationMaxMantissaBitCount(ctx->handle, 79);
  if (st != CUBLAS_STATUS_SUCCESS) {
    set_errf(err, errlen, "cublasSetFixedPointEmulationMaxMantissaBitCount failed", cublas_status_str(st));
    cudaFree(ctx->d_mantissa_bits);
    if (ctx->workspace) cudaFree(ctx->workspace);
    cublasDestroy(ctx->handle);
    delete ctx;
    return -11;
  }

  st = cublasSetFixedPointEmulationMantissaBitOffset(ctx->handle, 0);
  if (st != CUBLAS_STATUS_SUCCESS) {
    set_errf(err, errlen, "cublasSetFixedPointEmulationMantissaBitOffset failed", cublas_status_str(st));
    cudaFree(ctx->d_mantissa_bits);
    if (ctx->workspace) cudaFree(ctx->workspace);
    cublasDestroy(ctx->handle);
    delete ctx;
    return -12;
  }

  st = cublasSetFixedPointEmulationMantissaBitCountPointer(ctx->handle, ctx->d_mantissa_bits);
  if (st != CUBLAS_STATUS_SUCCESS) {
    set_errf(err, errlen, "cublasSetFixedPointEmulationMantissaBitCountPointer failed", cublas_status_str(st));
    cudaFree(ctx->d_mantissa_bits);
    if (ctx->workspace) cudaFree(ctx->workspace);
    cublasDestroy(ctx->handle);
    delete ctx;
    return -13;
  }

  *out_ctx = (void*)ctx;
  set_err(err, errlen, "");
  return 0;
}


OZAPI int ozaki_adp_destroy(void* pctx) {
  if (!pctx) return 0;
  OzakiCtx* ctx = (OzakiCtx*)pctx;
  if (ctx->handle) cublasDestroy(ctx->handle);
  if (ctx->d_mantissa_bits) cudaFree(ctx->d_mantissa_bits);
  if (ctx->workspace) cudaFree(ctx->workspace);
  delete ctx;
  return 0;
}


// IMPORTANT: cuBLAS assumes column-major. For row-major contiguous A,B,C (n x n),
// we compute C = A*B (row-major) by calling column-major GEMM with swapped A/B pointers:
//
// Let A_col = A_row^T, B_col = B_row^T (how cuBLAS "sees" row-major memory).
// If we compute C_col = B_col * A_col = (B_row^T)*(A_row^T) = (A_row*B_row)^T,
// then interpreting C_col memory as row-major yields (C_col)^T = A_row*B_row (desired).
//
OZAPI int ozaki_adp_gemm_ms(
    void* pctx,
    int n,
    const void* A_rowmajor,
    const void* B_rowmajor,
    void* C_rowmajor,
    int iters,
    int warmup,
    uint64_t stream_ptr,
    float* out_ms_per,
    int* out_mantissa_bits,
    char* err, int errlen)
{
  if (!pctx) { set_err(err, errlen, "ctx is null"); return -1; }
  if (!A_rowmajor || !B_rowmajor || !C_rowmajor) { set_err(err, errlen, "A/B/C is null"); return -2; }
  if (n <= 0) { set_err(err, errlen, "n <= 0"); return -3; }
  if (iters <= 0) iters = 1;
  if (warmup < 0) warmup = 0;

  OzakiCtx* ctx = (OzakiCtx*)pctx;

  cudaError_t ce = cudaSetDevice(ctx->device);
  if (ce != cudaSuccess) { set_errf(err, errlen, "cudaSetDevice failed", cudaGetErrorString(ce)); return -4; }

  cudaStream_t stream = (cudaStream_t)(uintptr_t)stream_ptr;
  cublasStatus_t st = cublasSetStream(ctx->handle, stream);
  if (st != CUBLAS_STATUS_SUCCESS) { set_errf(err, errlen, "cublasSetStream failed", cublas_status_str(st)); return -5; }

  // Re-apply workspace to be safe (some apps switch handles / configs).
  if (ctx->workspace && ctx->workspace_bytes > 0) {
    st = cublasSetWorkspace(ctx->handle, ctx->workspace, ctx->workspace_bytes);
    if (st != CUBLAS_STATUS_SUCCESS) {
      set_errf(err, errlen, "cublasSetWorkspace failed", cublas_status_str(st));
      return -6;
    }
  }

  const double alpha = 1.0;
  const double beta  = 0.0;

  // Swapped pointers: B then A (see note above).
  const void* Acol = B_rowmajor;
  const void* Bcol = A_rowmajor;
  void* Ccol = C_rowmajor;

  const int lda = n;
  const int ldb = n;
  const int ldc = n;

  // Warmup (not timed)
  for (int i = 0; i < warmup; ++i) {
    st = cublasGemmEx(ctx->handle,
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     n, n, n,
                     &alpha,
                     Acol, CUDA_R_64F, lda,
                     Bcol, CUDA_R_64F, ldb,
                     &beta,
                     Ccol, CUDA_R_64F, ldc,
                     CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT,
                     CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) {
      set_errf(err, errlen, "cublasGemmEx(warmup) failed", cublas_status_str(st));
      return -7;
    }
  }

  cudaEvent_t ev0, ev1;
  ce = cudaEventCreate(&ev0);
  if (ce != cudaSuccess) { set_errf(err, errlen, "cudaEventCreate ev0 failed", cudaGetErrorString(ce)); return -8; }
  ce = cudaEventCreate(&ev1);
  if (ce != cudaSuccess) { set_errf(err, errlen, "cudaEventCreate ev1 failed", cudaGetErrorString(ce)); cudaEventDestroy(ev0); return -9; }

  ce = cudaEventRecord(ev0, stream);
  if (ce != cudaSuccess) { set_errf(err, errlen, "cudaEventRecord ev0 failed", cudaGetErrorString(ce)); cudaEventDestroy(ev0); cudaEventDestroy(ev1); return -10; }

  for (int i = 0; i < iters; ++i) {
    st = cublasGemmEx(ctx->handle,
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     n, n, n,
                     &alpha,
                     Acol, CUDA_R_64F, lda,
                     Bcol, CUDA_R_64F, ldb,
                     &beta,
                     Ccol, CUDA_R_64F, ldc,
                     CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT,
                     CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) {
      set_errf(err, errlen, "cublasGemmEx failed", cublas_status_str(st));
      cudaEventDestroy(ev0); cudaEventDestroy(ev1);
      return -11;
    }
  }

  ce = cudaEventRecord(ev1, stream);
  if (ce != cudaSuccess) { set_errf(err, errlen, "cudaEventRecord ev1 failed", cudaGetErrorString(ce)); cudaEventDestroy(ev0); cudaEventDestroy(ev1); return -12; }

  ce = cudaEventSynchronize(ev1);
  if (ce != cudaSuccess) { set_errf(err, errlen, "cudaEventSynchronize failed", cudaGetErrorString(ce)); cudaEventDestroy(ev0); cudaEventDestroy(ev1); return -13; }

  float ms = 0.0f;
  ce = cudaEventElapsedTime(&ms, ev0, ev1);
  if (ce != cudaSuccess) { set_errf(err, errlen, "cudaEventElapsedTime failed", cudaGetErrorString(ce)); cudaEventDestroy(ev0); cudaEventDestroy(ev1); return -14; }

  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);

  // Copy mantissa bitcount back to host
  int host_bits = -1;
  ce = cudaMemcpyAsync(&host_bits, ctx->d_mantissa_bits, sizeof(int), cudaMemcpyDeviceToHost, stream);
  if (ce != cudaSuccess) { set_errf(err, errlen, "cudaMemcpyAsync(mantissa) failed", cudaGetErrorString(ce)); return -15; }
  ce = cudaStreamSynchronize(stream);
  if (ce != cudaSuccess) { set_errf(err, errlen, "cudaStreamSynchronize failed", cudaGetErrorString(ce)); return -16; }

  if (out_ms_per) *out_ms_per = ms / (float)iters;
  if (out_mantissa_bits) *out_mantissa_bits = host_bits;

  set_err(err, errlen, "");
  return 0;
}

} // extern "C"
"""


def _default_adp_workspace_bytes(n: int) -> int:
    """
    A pragmatic workspace sizing that matches typical fixed-point emulation scaling seen in practice.
    Chosen to be large enough to avoid internal errors and allow better kernels.

    Empirically fits:
      n=2048 -> ~239,095,808 bytes
      n=4096 -> ~553,668,608 bytes (~528 MiB)

    Formula: 25*n*n + 134,238,208
    """
    return int(25 * n * n + 134_238_208)


def _cache_dir() -> Path:
    # Local cache next to this script (portable, avoids user profile permission weirdness).
    here = Path(__file__).resolve().parent
    d = here / ".ozaki_adp_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _build_ozaki_adp_library(
    cc_major: int, cc_minor: int, verbose: bool = False
) -> Path:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError(
            "nvcc not found in PATH. Install CUDA Toolkit 13.1 and ensure nvcc is on PATH."
        )

    sysname = platform.system().lower()
    is_windows = sysname == "windows"

    cc = f"{cc_major}{cc_minor}"
    src = _CUDA_DDL_SOURCE
    key = hashlib.sha256((src + f"|sm{cc}|{sysname}").encode("utf-8")).hexdigest()[:16]

    out_dir = _cache_dir() / f"sm{cc}_{key}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cu_path = out_dir / "ozaki_adp_dll.cu"
    cu_path.write_text(src, encoding="utf-8")

    if is_windows:
        lib_path = out_dir / "ozaki_adp.dll"
    else:
        lib_path = out_dir / "libozaki_adp.so"

    if lib_path.exists():
        return lib_path

    # Build
    arch_flag = f"sm_{cc}"
    cmd: List[str] = [
        nvcc,
        "-O3",
        "-std=c++17",
        str(cu_path),
        "-shared",
        f"-arch={arch_flag}",
        "-lcublas",
        "-lcudart",
    ]

    if is_windows:
        # Keep MSVC runtime dynamic; --use-local-env helps find cl.exe if VS env is set.
        cmd.insert(1, "--use-local-env")
        cmd += ["-Xcompiler", "/MD"]
        cmd += ["-o", str(lib_path)]
    else:
        cmd += ["-Xcompiler", "-fPIC", "-o", str(lib_path)]

    if verbose:
        print("[build] " + " ".join(cmd))

    try:
        subprocess.check_call(cmd, cwd=str(out_dir))
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Failed to build Ozaki ADP DLL/SO with nvcc. "
            "On Windows you must run from a 'Developer Command Prompt for VS' so cl.exe/link.exe are available.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {e.returncode}"
        ) from e

    if not lib_path.exists():
        raise RuntimeError(
            "Build reported success but output library not found: " + str(lib_path)
        )

    return lib_path


class OzakiADP:
    def __init__(self, lib_path: Path):
        self.lib_path = lib_path
        self._dll = (
            ctypes.WinDLL(str(lib_path))
            if platform.system().lower() == "windows"
            else ctypes.CDLL(str(lib_path))
        )

        # int ozaki_adp_create(int device, size_t workspace_bytes, int emulation_strategy, char* err, int errlen, void** out_ctx)
        self._dll.ozaki_adp_create.argtypes = [
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._dll.ozaki_adp_create.restype = ctypes.c_int

        # int ozaki_adp_destroy(void* ctx)
        self._dll.ozaki_adp_destroy.argtypes = [ctypes.c_void_p]
        self._dll.ozaki_adp_destroy.restype = ctypes.c_int

        # int ozaki_adp_gemm_ms(void* ctx, int n, const void* A, const void* B, void* C,
        #                       int iters, int warmup, uint64_t stream_ptr,
        #                       float* out_ms_per, int* out_mantissa_bits,
        #                       char* err, int errlen)
        self._dll.ozaki_adp_gemm_ms.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        self._dll.ozaki_adp_gemm_ms.restype = ctypes.c_int

        self._ctx = ctypes.c_void_p(None)

    def create(
        self, device: int, workspace_bytes: int, emulation_strategy: int = 0
    ) -> None:
        errbuf = ctypes.create_string_buffer(2048)
        out = ctypes.c_void_p(None)
        rc = self._dll.ozaki_adp_create(
            int(device),
            int(workspace_bytes),
            int(emulation_strategy),
            errbuf,
            ctypes.sizeof(errbuf),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError(
                f"ozaki_adp_create failed rc={rc}: {errbuf.value.decode(errors='ignore')}"
            )
        self._ctx = out

    def destroy(self) -> None:
        if self._ctx and self._ctx.value:
            self._dll.ozaki_adp_destroy(self._ctx)
            self._ctx = ctypes.c_void_p(None)

    def gemm_ms(
        self,
        n: int,
        A_ptr: int,
        B_ptr: int,
        C_ptr: int,
        iters: int,
        warmup: int,
        stream_ptr: int,
    ) -> Tuple[float, int]:
        errbuf = ctypes.create_string_buffer(2048)
        out_ms = ctypes.c_float(0.0)
        out_bits = ctypes.c_int(-1)
        rc = self._dll.ozaki_adp_gemm_ms(
            self._ctx,
            int(n),
            ctypes.c_void_p(int(A_ptr)),
            ctypes.c_void_p(int(B_ptr)),
            ctypes.c_void_p(int(C_ptr)),
            int(iters),
            int(warmup),
            ctypes.c_uint64(int(stream_ptr)),
            ctypes.byref(out_ms),
            ctypes.byref(out_bits),
            errbuf,
            ctypes.sizeof(errbuf),
        )
        if rc != 0:
            raise RuntimeError(
                f"ozaki_adp_gemm_ms failed rc={rc}: {errbuf.value.decode(errors='ignore')}"
            )
        return float(out_ms.value), int(out_bits.value)

    def __enter__(self) -> "OzakiADP":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.destroy()


# =============================================================================
# Benchmarks
# =============================================================================


def bench_bandwidth(backend: Backend, n: int, repeat: int) -> Tuple[float, str]:
    """Device-to-device memcpy bandwidth."""
    if backend.is_torch:
        torch = backend.lib
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        fn = lambda: y.copy_(x)
    else:
        cp = backend.lib
        x = cp.random.random(n, dtype=cp.float32)
        y = cp.empty_like(x)

        copy_kernel = cp.RawKernel(
            r"""
            extern "C" __global__ void dcopy(const float* __restrict__ x, float* __restrict__ y, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int stride = gridDim.x * blockDim.x;
                for (int i = idx; i < n; i += stride) y[i] = x[i];
            }
            """,
            "dcopy",
        )
        threads = 256
        blocks = (n + threads - 1) // threads
        fn = lambda: copy_kernel((blocks,), (threads,), (x, y, n))

    sec = measure_gpu_seconds(backend, fn, repeat, warmup=3)
    bytes_total = 2 * n * 4
    gbps = bytes_total / sec / 1e9
    return gbps, "D2D memcpy (float32)"


def _validate_matrix_product(
    c_host: np.ndarray,
    ref_host: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Returns:
      abs_err_max
      rel_fro = ||E||_F / ||Ref||_F
      rel_err_max_scaled: max |E_ij| / max(|Ref_ij|, scale*1e-15), where scale=max|Ref|
    """
    diff = c_host - ref_host
    abs_max = float(np.max(np.abs(diff)))
    denom_fro = float(np.linalg.norm(ref_host))
    rel_fro = float(np.linalg.norm(diff) / (denom_fro if denom_fro != 0.0 else 1.0))

    scale = float(np.max(np.abs(ref_host)))
    floor = scale * 1e-15 if scale != 0.0 else 1e-15
    rel_max_scaled = float(np.max(np.abs(diff) / np.maximum(np.abs(ref_host), floor)))
    return abs_max, rel_fro, rel_max_scaled


def bench_tensor_core_gemm(
    backend: Backend,
    n: int,
    repeat: int,
    use_bf16: bool,
    validate_ozaki: str,
    adp_workspace_mib: int = 0,
    adp_verbose_build: bool = False,
) -> List[Tuple[str, float, str]]:
    results: List[Tuple[str, float, str]] = []
    lib = backend.lib

    dtypes: List[Tuple[str, Any]] = [
        ("FP16_TC", backend.float16),
    ]
    if use_bf16 and backend.bfloat16 is not None:
        dtypes.append(("BF16_TC", backend.bfloat16))
    dtypes.append(("TF32_TC", backend.float32))

    # Ozaki ADP (fixed-point FP64 emulation) benchmark label
    dtypes.append(("FP64_OZAKI_ADP", backend.float64))

    # Native FP64
    dtypes.append(("FP64", backend.float64))

    def _fp64_ozaki_adp_gemm() -> Tuple[str, float, str]:
        """
        FP64 emulation via cuBLAS fixed-point (Ozaki ADP).
        Implemented in a NVCC-built DLL/SO + ctypes call.
        """
        info = device_info(backend)
        cc_major, cc_minor = int(info["cc_major"]), int(info["cc_minor"])

        # Print env vars that can unexpectedly alter behavior
        env_keys = [
            "CUBLAS_EMULATION_STRATEGY",
            "CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT",
            "NVIDIA_TF32_OVERRIDE",
        ]
        for k in env_keys:
            if k in os.environ:
                print(f"ENV {k}={os.environ[k]}")

        lib_path = _build_ozaki_adp_library(
            cc_major, cc_minor, verbose=adp_verbose_build
        )

        # Workspace size (no trade-off: keep ADP dynamic; just provide workspace)
        if adp_workspace_mib > 0:
            workspace_bytes = int(adp_workspace_mib) * (1 << 20)
        else:
            workspace_bytes = _default_adp_workspace_bytes(n)

        # We keep emulation strategy = DEFAULT (0).
        # (If user set env var CUBLAS_EMULATION_STRATEGY, cuBLAS can override defaults.)
        emu_strategy = 0  # DEFAULT

        # Allocate A/B/C on GPU
        if backend.is_torch:
            torch = backend.lib
            A = torch.randn(n, n, device="cuda", dtype=torch.float64)
            B = torch.randn(n, n, device="cuda", dtype=torch.float64)
            C = torch.empty((n, n), device="cuda", dtype=torch.float64)
            A_ptr = int(A.data_ptr())
            B_ptr = int(B.data_ptr())
            C_ptr = int(C.data_ptr())
        else:
            cp = backend.lib
            A = cp.random.standard_normal((n, n), dtype=cp.float64)  # row-major
            B = cp.random.standard_normal((n, n), dtype=cp.float64)
            C = cp.empty((n, n), dtype=cp.float64)
            A_ptr = int(A.data.ptr)
            B_ptr = int(B.data.ptr)
            C_ptr = int(C.data.ptr)

        stream_ptr = _get_cuda_stream_ptr(backend)

        with OzakiADP(lib_path) as adp:
            # device id
            if backend.is_torch:
                dev_id = int(backend.lib.cuda.current_device())
            else:
                dev_id = int(backend.lib.cuda.Device().id)

            adp.create(
                device=dev_id,
                workspace_bytes=workspace_bytes,
                emulation_strategy=emu_strategy,
            )

            # Warmup + timed iters inside DLL (event-timed on the same stream)
            warmup = 3
            iters = max(1, repeat)
            ms_per, mantissa_bits = adp.gemm_ms(
                n=n,
                A_ptr=A_ptr,
                B_ptr=B_ptr,
                C_ptr=C_ptr,
                iters=iters,
                warmup=warmup,
                stream_ptr=stream_ptr,
            )

        # Convert to seconds and TFLOPS
        sec = ms_per / 1000.0
        tflops = 2.0 * n * n * n / sec / 1e12

        ws_mib = workspace_bytes / (1 << 20)

        rel_note = ""
        if validate_ozaki != "none":
            backend.synchronize()
            if backend.is_torch:
                A_host = A.detach().cpu().numpy()
                B_host = B.detach().cpu().numpy()
                C_host = C.detach().cpu().numpy()
            else:
                A_host = A.get()
                B_host = B.get()
                C_host = C.get()

            if validate_ozaki == "gpu":
                # GPU reference FP64
                if backend.is_torch:
                    torch = backend.lib
                    ref = (A @ B).detach().cpu().numpy()
                else:
                    cp = backend.lib
                    ref = (A @ B).get()
                abs_max, rel_fro, rel_max_scaled = _validate_matrix_product(C_host, ref)
                rel_note = f", abs_max={abs_max:.2e}, relFro={rel_fro:.2e}, relMax*={rel_max_scaled:.2e} (vs GPU FP64)"
            else:
                # CPU reference FP64
                ref = A_host @ B_host
                abs_max, rel_fro, rel_max_scaled = _validate_matrix_product(C_host, ref)
                rel_note = f", abs_max={abs_max:.2e}, relFro={rel_fro:.2e}, relMax*={rel_max_scaled:.2e} (vs CPU FP64)"

        notes = f"cuBLAS FP64 emu fixed-point (Ozaki ADP, mantissa_bits={mantissa_bits}, ws={ws_mib:.1f}MiB{rel_note})"
        return ("GEMM_FP64_OZAKI_ADP", tflops, notes)

    def _cupy_fp64_tc_gemm() -> Optional[Tuple[str, float, str]]:
        # Keep your previous "try FP64 tensor op path" attempt for CuPy.
        if backend.is_torch or backend.name != "cupy":
            return None
        cp = backend.lib
        try:
            handle = cp.cuda.get_cublas_handle()
            a = cp.random.standard_normal((n, n), dtype=cp.float64, order="F")
            b = cp.random.standard_normal((n, n), dtype=cp.float64, order="F")
            c = cp.zeros((n, n), dtype=cp.float64, order="F")
            alpha = np.float64(1.0)
            beta = np.float64(0.0)
            dtype = cp.cuda.cublas.CUDA_R_64F
            compute = getattr(
                cp.cuda.cublas,
                "CUBLAS_COMPUTE_64F_PEDANTIC",
                cp.cuda.cublas.CUBLAS_COMPUTE_64F,
            )
            algo = getattr(
                cp.cuda.cublas,
                "CUBLAS_GEMM_DEFAULT_TENSOR_OP",
                cp.cuda.cublas.CUBLAS_GEMM_DEFAULT,
            )

            def fn():
                cp.cuda.cublas.gemmEx(
                    handle,
                    cp.cuda.cublas.CUBLAS_OP_N,
                    cp.cuda.cublas.CUBLAS_OP_N,
                    n,
                    n,
                    n,
                    alpha.ctypes.data,
                    a.data.ptr,
                    dtype,
                    n,
                    b.data.ptr,
                    dtype,
                    n,
                    beta.ctypes.data,
                    c.data.ptr,
                    dtype,
                    n,
                    compute,
                    algo,
                )
                return c

            sec = measure_gpu_seconds(backend, fn, repeat)
            tflops = 2 * n * n * n / sec / 1e12
            return ("GEMM_FP64_TC", tflops, "FP64 cublasGemmEx (try Tensor Op algo)")
        except Exception:
            return None

    # Run GEMM suite
    for label, dtype in dtypes:
        if label == "FP64_OZAKI_ADP":
            results.append(_fp64_ozaki_adp_gemm())
            continue

        if label == "FP64" and not backend.is_torch and backend.name == "cupy":
            special = _cupy_fp64_tc_gemm()
            if special is not None:
                results.append(special)
                continue

        a = randn(backend, (n, n), dtype)
        b = randn(backend, (n, n), dtype)
        out = zeros(backend, (n, n), dtype)
        fn = lambda: matmul(backend, a, b, out=out)
        sec = measure_gpu_seconds(backend, fn, repeat)
        tflops = 2 * n * n * n / sec / 1e12

        if label == "FP64" and not backend.is_torch and backend.name == "cupy":
            notes = (
                "FP64 matmul (may fallback; TC FP64 path depends on hardware/library)"
            )
            name = "GEMM_FP64_TC"
        else:
            notes = f"{label} matmul (Tensor Core math where applicable)"
            name = f"GEMM_{label}"

        results.append((name, tflops, notes))
        del out
        if hasattr(lib, "cuda"):
            backend.synchronize()

    return results


def bench_tebd_two_site(
    backend: Backend, bond: int, phys_dim: int, batch: int, repeat: int
) -> Tuple[float, str]:
    dtype = backend.float32
    A = randn(backend, (batch, bond, phys_dim, bond), dtype)
    B = randn(backend, (batch, bond, phys_dim, bond), dtype)
    G = randn(backend, (phys_dim, phys_dim, phys_dim, phys_dim), dtype)

    def fn():
        return einsum(backend, "baim,bmjc,ijpq->bapqc", A, B, G)

    sec = measure_gpu_seconds(backend, fn, repeat)
    flops = 2 * batch * (bond**2) * (bond) * (phys_dim**2)
    tflops = flops / sec / 1e12
    return tflops, f"Two-site TEBD contraction (bond={bond}, phys={phys_dim})"


def bench_mpo_env(
    backend: Backend, bond: int, phys_dim: int, batch: int, repeat: int
) -> Tuple[float, str]:
    dtype = backend.float32
    L = randn(backend, (batch, bond, bond), dtype)
    W = randn(backend, (batch, bond, phys_dim, bond, phys_dim), dtype)
    A = randn(backend, (batch, bond, phys_dim), dtype)

    def fn():
        tmp = einsum(backend, "blm,blxry->bmxry", L, W)
        return einsum(backend, "bmxry,bmx,bny->brn", tmp, A, A)

    sec = measure_gpu_seconds(backend, fn, repeat)
    flops = 2 * batch * (bond**3) * (phys_dim**2)
    tflops = flops / sec / 1e12
    return tflops, f"MPO/MPS env contraction (bond={bond}, phys={phys_dim})"


def bench_truncated_svd(backend: Backend, dim: int, repeat: int) -> Tuple[float, str]:
    dtype = backend.float32
    X = randn(backend, (dim, dim), dtype)

    def fn():
        return svd(backend, X)

    sec = measure_gpu_seconds(backend, fn, repeat=repeat, warmup=1)
    flops = 4.0 / 3.0 * (dim**3)
    tflops = flops / sec / 1e12
    return tflops, f"SVD (dim={dim})"


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CUDA 13.1 Tensor-Network benchmark + Ozaki ADP (DLL+ctypes)"
    )
    p.add_argument("--backend", default="auto", choices=["auto", "cupy", "torch"])
    p.add_argument(
        "--matrix-size", type=int, default=4096, help="Matrix size for GEMM benchmarks."
    )
    p.add_argument(
        "--bond", type=int, default=512, help="Bond dimension for MPS tests."
    )
    p.add_argument(
        "--phys-dim", type=int, default=4, help="Physical dimension for MPS tests."
    )
    p.add_argument(
        "--batch", type=int, default=8, help="Number of parallel MPS chains."
    )
    p.add_argument(
        "--svd-dim", type=int, default=2048, help="Matrix dimension for SVD benchmark."
    )
    p.add_argument("--repeat", type=int, default=10, help="Repetitions per benchmark.")
    p.add_argument(
        "--bf16",
        action="store_true",
        help="Include BF16 GEMM if the backend supports it.",
    )
    p.add_argument(
        "--validate-ozaki",
        default="none",
        choices=["none", "gpu", "cpu"],
        help="Validate Ozaki ADP result against GPU FP64 or CPU FP64 (adds overhead).",
    )
    p.add_argument(
        "--adp-workspace-mib",
        type=int,
        default=0,
        help="Override Ozaki ADP workspace (MiB). 0 = auto.",
    )
    p.add_argument(
        "--adp-verbose-build",
        action="store_true",
        help="Print nvcc build command for the Ozaki ADP DLL/SO.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    backend = choose_backend(args.backend)
    info = device_info(backend)

    print(f"Backend     : {backend.name}")
    print(f"GPU         : {info['name']} (CC {info['cc']}, SMs={info.get('sm_count')})")
    print(f"Memory      : {info['total_mem_gb']:.2f} GB")
    print("Math mode   : Tensor Core enabled where supported (CUDA 13.1)")
    print(
        "Ozaki ADP   : cuBLAS fixed-point FP64 emulation (CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT)\n"
    )

    results: List[Tuple[str, float, str, str]] = []

    # Bandwidth
    gbps, notes = bench_bandwidth(
        backend, n=256 * 1024 * 1024 // 4, repeat=max(3, args.repeat // 2)
    )
    results.append(("Bandwidth", gbps, "GB/s", notes))

    # GEMM suite (includes Ozaki ADP)
    for name, tflops, notes in bench_tensor_core_gemm(
        backend,
        n=args.matrix_size,
        repeat=args.repeat,
        use_bf16=args.bf16,
        validate_ozaki=args.validate_ozaki,
        adp_workspace_mib=args.adp_workspace_mib,
        adp_verbose_build=args.adp_verbose_build,
    ):
        results.append((name, tflops, "TFLOPS", notes))

    # TEBD two-site
    tebd_tflops, notes = bench_tebd_two_site(
        backend,
        bond=args.bond,
        phys_dim=args.phys_dim,
        batch=args.batch,
        repeat=args.repeat,
    )
    results.append(("TEBD_2site", tebd_tflops, "TFLOPS", notes))

    # MPO/MPS environment
    env_tflops, notes = bench_mpo_env(
        backend,
        bond=args.bond,
        phys_dim=args.phys_dim,
        batch=args.batch,
        repeat=args.repeat,
    )
    results.append(("MPO_env", env_tflops, "TFLOPS", notes))

    # Truncated SVD
    svd_tflops, notes = bench_truncated_svd(
        backend, dim=args.svd_dim, repeat=max(3, args.repeat // 2)
    )
    results.append(("SVD_trunc", svd_tflops, "TFLOPS", notes))

    print("Benchmark results:")
    for name, metric, unit, notes in results:
        print(format_row(name, metric, unit, notes))

    print(
        "\nNotes:\n"
        "- GEMM_FP64_OZAKI_ADP uses cuBLAS fixed-point FP64 emulation + ADP; mantissa bits are device-written.\n"
        "- Validation prints abs_max, relFro, relMax* (scaled) to avoid near-zero ref entries dominating.\n"
        "- TEBD_2site and MPO_env use einsum-based contractions representative of MPS workloads.\n"
        "- SVD_trunc mirrors the cost of two-site truncations; increase --svd-dim for larger bonds.\n"
        "- If you set env var CUBLAS_EMULATION_STRATEGY / CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT, it may change behavior."
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
