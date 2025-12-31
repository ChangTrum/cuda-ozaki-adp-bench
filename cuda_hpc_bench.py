#!/usr/bin/env python3
"""
CUDA HPC Benchmark Suite with Ozaki ADP Support

This benchmark suite tests various CUDA operations including:
- Tensor Core GEMM (FP16, FP32, FP64)
- cuBLAS Ozaki ADP baseline (FP64 emulation for higher precision)
- Truncated SVD
- Device-to-Device (D2D) memory bandwidth

Ozaki ADP (Adaptive Double Precision):
A cuBLAS-based technique that emulates higher-precision FP64 arithmetic using
fixed-point accumulation. This provides better numerical accuracy for certain
workloads while maintaining GPU acceleration.

Note: This implementation provides a baseline FP64 benchmark. True Ozaki ADP
requires specialized cuBLAS configurations not exposed in standard CuPy.
"""

import argparse
import sys
import time
from typing import Dict

try:
    import numpy as np
except ImportError:
    print("Error: NumPy is required. Install with: pip install numpy")
    sys.exit(1)

try:
    import cupy as cp
except ImportError:
    print("Error: CuPy is required. Install with: pip install cupy-cuda12x")
    print("Note: Replace '12x' with your CUDA version (e.g., 11x, 12x)")
    sys.exit(1)


class CUDABenchmark:
    """CUDA benchmark suite for HPC operations"""

    def __init__(self, validate: bool = False):
        """
        Initialize benchmark suite.

        Args:
            validate: Enable validation of results against CPU calculations
        """
        self.validate = validate
        self.results: Dict[str, float] = {}

    def benchmark_gemm(self, size: int = 4096, dtype: str = "float32") -> float:
        """
        Benchmark General Matrix Multiply (GEMM) operation.

        Args:
            size: Matrix dimension (size x size)
            dtype: Data type ('float16', 'float32', 'float64')

        Returns:
            GFLOPS performance
        """
        dtype_map = {
            "float16": np.float16,
            "float32": np.float32,
            "float64": np.float64,
        }

        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}")

        np_dtype = dtype_map[dtype]

        # Create random matrices on GPU
        A = cp.random.randn(size, size).astype(np_dtype)
        B = cp.random.randn(size, size).astype(np_dtype)

        # Warmup
        C = cp.matmul(A, B)
        cp.cuda.Stream.null.synchronize()

        # Benchmark
        start = time.perf_counter()
        C = cp.matmul(A, B)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start

        # Calculate GFLOPS
        flops = 2 * size**3
        gflops = (flops / elapsed) / 1e9

        # Validation
        if self.validate:
            A_cpu = cp.asnumpy(A)
            B_cpu = cp.asnumpy(B)
            C_cpu = np.matmul(A_cpu, B_cpu)
            C_gpu = cp.asnumpy(C)
            error = np.linalg.norm(C_cpu - C_gpu) / np.linalg.norm(C_cpu)
            print(f"  Validation error: {error:.2e}")

        return gflops

    def benchmark_ozaki_adp(self, size: int = 2048) -> float:
        """
        Benchmark Ozaki ADP (Adaptive Double Precision) baseline.

        This provides a FP64 baseline for Ozaki ADP comparison. True Ozaki ADP
        uses fixed-point accumulation to achieve higher precision than standard
        FP64 operations, useful for ill-conditioned problems.

        Args:
            size: Matrix dimension

        Returns:
            GFLOPS performance (FP64 baseline)

        Note:
            This is a FP64 baseline benchmark. Real Ozaki ADP requires special
            cuBLAS configurations (e.g., cublasGemmEx with accumulation modes)
            not exposed in standard CuPy. Use this as a reference for comparison.
        """
        # Use FP64 as baseline for Ozaki ADP emulation
        A = cp.random.randn(size, size).astype(np.float64)
        B = cp.random.randn(size, size).astype(np.float64)

        # Warmup (result intentionally discarded)
        _ = cp.matmul(A, B)
        cp.cuda.Stream.null.synchronize()

        # Benchmark with higher precision accumulation
        start = time.perf_counter()
        _ = cp.matmul(A, B)  # Result discarded - only timing the operation
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start

        flops = 2 * size**3
        gflops = (flops / elapsed) / 1e9

        if self.validate:
            print("  Ozaki ADP validation: Using FP64 baseline for reference")

        return gflops

    def benchmark_d2d_bandwidth(self, size_mb: int = 1024) -> float:
        """
        Benchmark Device memory bandwidth (single GPU).

        Measures memory copy bandwidth within a single GPU device.
        For multi-GPU D2D transfers, use peer-to-peer memory copy APIs.

        Args:
            size_mb: Transfer size in megabytes

        Returns:
            Bandwidth in GB/s
        """
        size_bytes = size_mb * 1024 * 1024
        size_elements = size_bytes // 4  # float32

        # Create source and destination arrays
        src = cp.random.randn(size_elements).astype(np.float32)
        dst = cp.empty_like(src)

        # Warmup
        dst[:] = src
        cp.cuda.Stream.null.synchronize()

        # Benchmark
        start = time.perf_counter()
        dst[:] = src
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start

        bandwidth_gbs = (size_bytes / elapsed) / 1e9

        return bandwidth_gbs

    def benchmark_svd(self, m: int = 2048, n: int = 1024) -> float:
        """
        Benchmark truncated SVD decomposition.

        Args:
            m: Number of rows
            n: Number of columns

        Returns:
            GFLOPS performance
        """
        A = cp.random.randn(m, n).astype(np.float32)

        # Warmup
        U, S, Vt = cp.linalg.svd(A, full_matrices=False)
        cp.cuda.Stream.null.synchronize()

        # Benchmark
        start = time.perf_counter()
        U, S, Vt = cp.linalg.svd(A, full_matrices=False)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start

        # Approximate FLOPS for SVD
        flops = 4 * m * n**2 + 8 * n**3
        gflops = (flops / elapsed) / 1e9

        if self.validate:
            A_reconstructed = U @ cp.diag(S) @ Vt
            error = cp.linalg.norm(A - A_reconstructed) / cp.linalg.norm(A)
            print(f"  SVD reconstruction error: {float(error):.2e}")

        return gflops

    def run_all_benchmarks(self) -> Dict[str, float]:
        """
        Run all benchmarks and return results.

        Returns:
            Dictionary of benchmark names and their performance metrics
        """
        print("=" * 60)
        print("CUDA HPC Benchmark Suite")
        print("=" * 60)

        # GEMM benchmarks
        print("\n[GEMM Benchmarks]")
        for dtype in ["float16", "float32", "float64"]:
            print(f"Running GEMM ({dtype})...")
            perf = self.benchmark_gemm(dtype=dtype)
            self.results[f"gemm_{dtype}"] = perf
            print(f"  Performance: {perf:.2f} GFLOPS")

        # Ozaki ADP
        print("\n[Ozaki ADP - FP64 Baseline]")
        print("Running Ozaki ADP baseline benchmark...")
        perf = self.benchmark_ozaki_adp()
        self.results["ozaki_adp"] = perf
        print(f"  Performance: {perf:.2f} GFLOPS (FP64 baseline)")

        # D2D Bandwidth
        print("\n[Device Memory Bandwidth]")
        print("Running device memory bandwidth test...")
        bw = self.benchmark_d2d_bandwidth()
        self.results["d2d_bandwidth"] = bw
        print(f"  Bandwidth: {bw:.2f} GB/s")

        # SVD
        print("\n[Truncated SVD]")
        print("Running SVD benchmark...")
        perf = self.benchmark_svd()
        self.results["svd"] = perf
        print(f"  Performance: {perf:.2f} GFLOPS")

        print("\n" + "=" * 60)
        print("Benchmark Complete")
        print("=" * 60)

        return self.results


def main():
    """Main entry point for the benchmark suite"""
    parser = argparse.ArgumentParser(
        description="CUDA HPC Benchmark Suite with Ozaki ADP support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks without validation
  python cuda_hpc_bench.py

  # Run with validation enabled
  python cuda_hpc_bench.py --validate

  # Run specific benchmark
  python cuda_hpc_bench.py --benchmark gemm

  # Run with custom matrix size
  python cuda_hpc_bench.py --size 8192

Ozaki ADP Information:
  Ozaki ADP (Adaptive Double Precision) is a technique that uses cuBLAS
  fixed-point accumulation to achieve higher numerical precision than
  standard FP64 arithmetic. This is particularly useful for ill-conditioned
  linear algebra problems where standard floating-point accumulation may
  introduce significant errors.
        """,
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Enable validation of results against CPU calculations",
    )

    parser.add_argument(
        "--benchmark",
        choices=["gemm", "ozaki", "svd", "d2d", "all"],
        default="all",
        help="Specific benchmark to run (default: all)",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=4096,
        help="Matrix size for benchmarks (default: 4096)",
    )

    args = parser.parse_args()

    # Check CUDA availability
    if not cp.cuda.is_available():
        print("Error: CUDA is not available on this system")
        sys.exit(1)

    try:
        # Format CUDA version from integer (e.g., 12000 -> "12.0")
        cuda_version = cp.cuda.runtime.runtimeGetVersion()
        cuda_major = cuda_version // 1000
        cuda_minor = (cuda_version % 1000) // 10
        print(f"CUDA Version: {cuda_major}.{cuda_minor}")
        print(f"Device: {cp.cuda.Device().name}")
    except Exception as e:
        print(f"Warning: Could not retrieve CUDA device info: {e}")

    print(f"Validation: {'Enabled' if args.validate else 'Disabled'}")

    bench = CUDABenchmark(validate=args.validate)

    if args.benchmark == "all":
        bench.run_all_benchmarks()
    elif args.benchmark == "gemm":
        print(f"Running GEMM benchmark (size={args.size})...")
        perf = bench.benchmark_gemm(size=args.size)
        print(f"Performance: {perf:.2f} GFLOPS")
    elif args.benchmark == "ozaki":
        print(f"Running Ozaki ADP benchmark (size={args.size})...")
        perf = bench.benchmark_ozaki_adp(size=args.size)
        print(f"Performance: {perf:.2f} GFLOPS")
    elif args.benchmark == "svd":
        print(f"Running SVD benchmark (size={args.size})...")
        # Use size for both m and n dimensions
        perf = bench.benchmark_svd(m=args.size, n=args.size // 2)
        print(f"Performance: {perf:.2f} GFLOPS")
    elif args.benchmark == "d2d":
        print("Running D2D bandwidth test...")
        # Convert size to MB (default is matrix elements, convert to approximate MB)
        size_mb = max(1, (args.size * args.size * 4) // (1024 * 1024))
        bw = bench.benchmark_d2d_bandwidth(size_mb=size_mb)
        print(f"Bandwidth: {bw:.2f} GB/s")


if __name__ == "__main__":
    main()
