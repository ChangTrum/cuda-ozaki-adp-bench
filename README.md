# cuda-ozaki-adp-bench

Single-file CUDA benchmark suite for HPC operations with Ozaki ADP support.

## Overview

This repository contains a minimal Python benchmark suite for CUDA operations, including:
- **Tensor Core GEMM**: Matrix multiplication with FP16, FP32, and FP64 precision
- **Ozaki ADP**: cuBLAS fixed-point FP64 emulation for higher numerical accuracy
- **Truncated SVD**: Singular Value Decomposition
- **D2D Bandwidth**: Device-to-Device memory transfer benchmarks

## What is Ozaki ADP?

**Ozaki ADP (Adaptive Double Precision)** is a technique that uses cuBLAS fixed-point accumulation to achieve higher numerical precision than standard FP64 arithmetic. This method is particularly valuable for:
- Ill-conditioned linear algebra problems
- Applications requiring extended precision without resorting to software-based arbitrary precision
- Scenarios where standard floating-point accumulation introduces significant rounding errors

The technique leverages CUDA's hardware capabilities to perform accumulation with fixed-point arithmetic, providing better accuracy while maintaining GPU acceleration.

## Requirements

### System Requirements
- NVIDIA GPU with CUDA support (Compute Capability 7.0+)
- CUDA Toolkit 11.x or 12.x
- Linux, Windows, or macOS with CUDA support

### Python Requirements
- Python 3.8+
- NumPy
- CuPy (matching your CUDA version)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/ChangTrum/cuda-ozaki-adp-bench.git
cd cuda-ozaki-adp-bench
```

2. Install Python dependencies:
```bash
# For CUDA 12.x
pip install numpy cupy-cuda12x

# For CUDA 11.x
pip install numpy cupy-cuda11x
```

## Usage

### Basic Usage

Run all benchmarks:
```bash
python cuda_hpc_bench.py
```

### Validation Options

Enable result validation against CPU calculations:
```bash
python cuda_hpc_bench.py --validate
```

This will compute reference results on the CPU and compare them with GPU results, reporting numerical errors.

### Running Specific Benchmarks

Run only GEMM benchmark:
```bash
python cuda_hpc_bench.py --benchmark gemm
```

Run only Ozaki ADP benchmark:
```bash
python cuda_hpc_bench.py --benchmark ozaki
```

Available benchmarks: `gemm`, `ozaki`, `svd`, `d2d`, `all`

### Custom Matrix Sizes

Specify custom matrix dimensions:
```bash
python cuda_hpc_bench.py --size 8192
```

### Full Example

Run GEMM benchmark with validation and custom size:
```bash
python cuda_hpc_bench.py --benchmark gemm --size 8192 --validate
```

## Command-Line Options

```
usage: cuda_hpc_bench.py [-h] [--validate] [--benchmark {gemm,ozaki,svd,d2d,all}] [--size SIZE]

Options:
  --validate          Enable validation of results against CPU calculations
  --benchmark BENCH   Specific benchmark to run: gemm, ozaki, svd, d2d, all (default: all)
  --size SIZE        Matrix size for benchmarks (default: 4096)
```

## Benchmark Details

### GEMM (General Matrix Multiply)
Tests matrix multiplication performance across different precisions (FP16, FP32, FP64). Reports performance in GFLOPS.

### Ozaki ADP
Evaluates fixed-point FP64 emulation performance. Uses cuBLAS-based techniques for higher numerical accuracy.

### SVD (Singular Value Decomposition)
Benchmarks truncated SVD decomposition, commonly used in tensor networks and data compression.

### D2D Bandwidth
Measures Device-to-Device memory transfer bandwidth in GB/s.

## Output Interpretation

The benchmark suite reports:
- **GFLOPS** (Giga Floating Point Operations Per Second) for compute operations
- **GB/s** (Gigabytes per second) for bandwidth tests
- **Validation errors** (when `--validate` is enabled) as relative errors

## Development

### Linting

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting. To check code quality:

```bash
pip install ruff
ruff check cuda_hpc_bench.py
```

### CI/CD

GitHub Actions automatically runs Ruff linting on all pushes and pull requests.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please ensure:
1. Code passes Ruff linting
2. Changes are tested on CUDA hardware
3. Documentation is updated for new features
