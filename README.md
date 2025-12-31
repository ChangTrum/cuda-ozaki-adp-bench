# CUDA 13.x Tensor-Network Benchmarks + Ozaki ADP (cuBLAS FP64 Emulation)

Single-file CUDA benchmark suite aimed at Blackwell-class systems (CC 10.x–12.x), with an FP64-emulation GEMM path using cuBLAS fixed-point (“Ozaki scheme”) + ADP (automatic dynamic precision) through a runtime-built NVCC shared library (DLL/SO) invoked via `ctypes`.

Benchmarks included:
- Tensor Core GEMM throughput: FP16 / BF16 (optional) / TF32 / FP64
- FP64 emulation GEMM: cuBLAS fixed-point + ADP (Ozaki ADP), timed inside the DLL/SO with CUDA events
- TEBD-style two-site MPS gate contraction (`einsum`)
- MPO/MPS environment contraction (`einsum`)
- Truncated SVD workload (`svd(full_matrices=False)`)
- Device-to-device bandwidth (D2D copy)

The goal is quick, reproducible “how fast is this GPU + stack for the stuff I care about?” runs—without needing a multi-file build system.

---

## Repository Layout

- `cuda_hpc_bench.py` — the only script you need.

---

## Requirements

Hardware:
- NVIDIA GPU with CUDA support (optimized for modern Tensor Core GPUs; CC 10.x–12.x is the intended target).

Software:
- Python 3.9+ (3.10+ recommended)
- NVIDIA driver compatible with CUDA 13.x
- CUDA Toolkit 13.x installed (needed for `nvcc` and runtime libs)
- One GPU Python backend:
  - **CuPy** (recommended), or
  - **PyTorch** with CUDA support

For the **Ozaki ADP** FP64 emulation benchmark (runtime NVCC build):
- `nvcc` must be on `PATH`
- A host compiler toolchain must be available:
  - Linux: `g++` (build-essential)
  - Windows: run in “Developer Command Prompt for VS” (so `cl.exe` / `link.exe` exist)

---

## Installation

### Option A: CuPy (recommended)

    pip install -U numpy cupy-cuda13x

### Option B: PyTorch

Install a CUDA-enabled PyTorch build appropriate for your environment (use PyTorch’s official install selector), then:

    pip install -U numpy

---

## Quick Start

Run with defaults (auto backend, GEMM n=4096, repeat=10):

    python cuda_hpc_bench.py

Force CuPy:

    python cuda_hpc_bench.py --backend cupy

Force PyTorch:

    python cuda_hpc_bench.py --backend torch

Include BF16 GEMM (if supported by your backend/GPU):

    python cuda_hpc_bench.py --bf16

Larger GEMM and more repeats:

    python cuda_hpc_bench.py --matrix-size 8192 --repeat 20

Validate Ozaki ADP against **GPU FP64** reference (adds overhead, ref stays on GPU):

    python cuda_hpc_bench.py --validate-ozaki gpu

Validate Ozaki ADP against **CPU FP64** reference (heavier: transfers A/B and runs NumPy matmul):

    python cuda_hpc_bench.py --validate-ozaki cpu

Print the `nvcc` command used to build the Ozaki DLL/SO:

    python cuda_hpc_bench.py --adp-verbose-build

Override Ozaki workspace (MiB):

    python cuda_hpc_bench.py --adp-workspace-mib 2048

---

## What Each Benchmark Measures

### 1) Bandwidth (D2D)

A simple device-to-device copy to sanity check that memory copy throughput is in the expected ballpark.

### 2) GEMM Suite

Dense `n × n` matmul for:
- FP16 Tensor Core
- BF16 Tensor Core (optional)
- TF32 Tensor Core (TF32 math enabled where available)
- FP64 native
- FP64 emulation (Ozaki ADP): cuBLAS fixed-point FP64 emulation via a runtime-built shared library

Throughput is reported as TFLOPS using `2*n^3 / time`.

Notes:
- On PyTorch, TF32 is enabled via `torch.backends.cuda.matmul.allow_tf32 = True` and `torch.set_float32_matmul_precision("high")` (best-effort).
- On CuPy, the script attempts to set cuBLAS math mode to tensor-op math mode (best-effort).
- FP64 “Tensor Op” behavior depends on GPU + CUDA/cuBLAS support. The script includes a CuPy-only best-effort attempt to call `cublasGemmEx` with a Tensor Op algorithm for FP64.

### 3) TEBD-style Two-Site Contraction

An `einsum` shaped like a two-site gate application in MPS/TEBD workflows:

    "baim,bmjc,ijpq->bapqc"

### 4) MPO/MPS Environment Contraction

A two-stage contraction representative of environment builds / effective Hamiltonian application patterns.

### 5) Truncated SVD Proxy

Runs `svd(full_matrices=False)` on a square matrix. Useful as a rough proxy for truncation-heavy steps (e.g., two-site updates).

---

## Ozaki ADP (FP64 Emulation) Details

This benchmark uses cuBLAS fixed-point FP64 emulation plus ADP (automatic dynamic precision).

How it works in this script:
- A small C++/CUDA shim is embedded as a string and compiled at runtime by `nvcc` into:
  - Windows: `ozaki_adp.dll`
  - Linux: `libozaki_adp.so`
- Python loads it via `ctypes` and calls `ozaki_adp_gemm_ms(...)`.

Key cuBLAS configuration performed inside the library:
- Uses `cublasGemmEx` with compute type:
  - `CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT`
- ADP (dynamic mantissa) is enabled and configured:
  - mantissa control: dynamic
  - max mantissa bits: 79
  - mantissa bit offset: 0
  - mantissa bit count writeback pointer: **device pointer** (important; host pointer can trigger failures)

Timing:
- The DLL/SO creates CUDA events on the provided stream and returns `ms_per_iter`.

Row-major vs column-major:
- cuBLAS assumes column-major. The library expects `A`, `B`, `C` to be **row-major contiguous** and uses a standard trick:
  - interpret row-major memory as transposed column-major, then swap A/B pointers to obtain the correct result.

Workspace:
- Fixed-point emulation can require significant workspace. The library uses `cublasSetWorkspace`.
- Default auto sizing (if `--adp-workspace-mib` is not set):

    workspace_bytes = 25*n*n + 134,238,208

Example:
- `n=4096` → ~528 MiB

If you see cuBLAS internal errors or poor performance, increase workspace:

    python cuda_hpc_bench.py --adp-workspace-mib 4096

---

## Ozaki ADP Validation Metrics

When `--validate-ozaki` is enabled, the script reports:

- `abs_max`:
  - maximum absolute error: `max_ij |C_ij - Ref_ij|`
- `relFro`:
  - relative Frobenius error: `||C-Ref||_F / ||Ref||_F`
- `relMax*` (scaled max relative error):
  - avoids near-zero reference entries dominating the max relative error:
    - `max_ij |E_ij| / max(|Ref_ij|, scale*1e-15)`
    - where `scale = max_ij |Ref_ij|`

Validation modes:
- `--validate-ozaki gpu`:
  - reference is GPU native FP64 matmul
- `--validate-ozaki cpu`:
  - reference is CPU FP64 (`numpy`), requires transferring A/B to host and is much heavier

---

## CLI Options

    --backend {auto,cupy,torch}     Backend selection (default: auto)
    --matrix-size N                 GEMM matrix size (default: 4096)
    --repeat R                      Repetitions per benchmark (default: 10)

    --bf16                          Include BF16 GEMM if supported

    --bond D                        Bond dimension for MPS tests (default: 512)
    --phys-dim P                    Physical dimension (default: 4)
    --batch B                       Number of parallel MPS chains (default: 8)

    --svd-dim S                     SVD dimension (default: 2048)

    --validate-ozaki {none,gpu,cpu} Validate Ozaki result (default: none)

    --adp-workspace-mib MiB         Override Ozaki workspace (0 = auto)
    --adp-verbose-build             Print nvcc build command

---

## Example Output (shape)

    Backend     : cupy
    GPU         : NVIDIA ... (CC 12.0, SMs=...)
    Memory      : 95.00 GB
    Math mode   : Tensor Core enabled where supported (CUDA 13.1)
    Ozaki ADP   : cuBLAS fixed-point FP64 emulation (CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT)

    Benchmark results:
    Bandwidth                     1500.00 GB/s   D2D memcpy (float32)
    GEMM_FP16_TC                   900.00 TFLOPS  FP16_TC matmul (Tensor Core math where applicable)
    GEMM_BF16_TC                   850.00 TFLOPS  BF16_TC matmul (Tensor Core math where applicable)
    GEMM_TF32_TC                   200.00 TFLOPS  TF32_TC matmul (Tensor Core math where applicable)
    GEMM_FP64_OZAKI_ADP            120.00 TFLOPS  cuBLAS FP64 emu fixed-point (Ozaki ADP, mantissa_bits=..., ws=...MiB, ...)
    GEMM_FP64                       40.00 TFLOPS  FP64 matmul
    TEBD_2site                      15.00 TFLOPS  Two-site TEBD contraction (bond=..., phys=...)
    MPO_env                         10.00 TFLOPS  MPO/MPS env contraction (bond=..., phys=...)
    SVD_trunc                        5.00 TFLOPS  SVD (dim=...)

---

## Build + Cache Behavior (Ozaki DLL/SO)

- The runtime-built library is cached next to the script:

    ./.ozaki_adp_cache/

- Cache key includes:
  - the embedded CUDA source hash
  - detected SM architecture (`sm_{cc}`)
  - OS (`windows` vs others)

To force a rebuild, delete the cache directory:

    rm -rf .ozaki_adp_cache

---

## Environment Variables (FYI)

The script prints these if present because they can change behavior:

- `CUBLAS_EMULATION_STRATEGY`
- `CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT`
- `NVIDIA_TF32_OVERRIDE`

---

## Troubleshooting

### `nvcc` not found
- Install CUDA Toolkit 13.x and ensure `nvcc` is on `PATH`.

### Windows build fails (cl/link not found)
- Run inside “Developer Command Prompt for VS” (or ensure MSVC Build Tools are installed and discoverable).

### cuBLAS errors during FP64 emulation (e.g., internal error / execution failed)
- Increase workspace:
  
      python cuda_hpc_bench.py --adp-workspace-mib 4096

- Confirm your driver + toolkit are consistent for CUDA 13.x.

### CuPy imports but CUDA libs fail at runtime
- Ensure CUDA runtime libraries are discoverable:
  - Linux: `LD_LIBRARY_PATH`
  - Windows: `PATH`

---

## License

Add whichever license you want (MIT/BSD/Apache-2.0/etc) by adding a `LICENSE` file at the repository root.
