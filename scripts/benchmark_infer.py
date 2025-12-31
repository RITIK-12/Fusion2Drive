#!/usr/bin/env python3
"""
Benchmark model inference speed on different platforms.

Usage:
    python scripts/benchmark_infer.py --checkpoint checkpoints/best.pt --device cuda
    python scripts/benchmark_infer.py --checkpoint checkpoints/best.pt --device mps
    python scripts/benchmark_infer.py --exported exports/model_fusion.onnx --format onnx
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wod_fusion.models import FusionModel, FusionModelConfig


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def create_dummy_inputs(batch_size: int = 1, device: str = "cpu"):
    """Create dummy inputs for benchmarking."""
    return {
        "images": torch.randn(batch_size, 5, 3, 256, 704, device=device),
        "intrinsics": torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, 5, -1, -1),
        "extrinsics": torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, 5, -1, -1),
        "points": torch.randn(batch_size, 150000, 4, device=device),
        "points_mask": torch.ones(batch_size, 150000, dtype=torch.bool, device=device),
    }


def benchmark_pytorch(
    model: FusionModel,
    device: str,
    batch_size: int = 1,
    num_warmup: int = 10,
    num_iterations: int = 100,
):
    """Benchmark PyTorch model."""
    logger = logging.getLogger(__name__)
    
    model = model.to(device)
    model.eval()
    
    inputs = create_dummy_inputs(batch_size, device)
    
    # Warmup
    logger.info("Warming up...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model.inference(**inputs)
    
    # Synchronize if CUDA
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    logger.info(f"Running {num_iterations} iterations...")
    latencies = []
    
    with torch.no_grad():
        for _ in range(num_iterations):
            if device == "cuda":
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model.inference(**inputs)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            latencies.append((time.perf_counter() - start) * 1000)
    
    return {
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
        "p50_ms": np.percentile(latencies, 50),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "fps": 1000 / np.mean(latencies),
    }


def benchmark_onnx(
    model_path: str,
    batch_size: int = 1,
    num_warmup: int = 10,
    num_iterations: int = 100,
):
    """Benchmark ONNX model."""
    import onnxruntime as ort
    
    logger = logging.getLogger(__name__)
    
    # Create session with optimizations
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    
    logger.info(f"ONNX providers: {session.get_providers()}")
    
    # Create inputs
    inputs = {
        "images": np.random.randn(batch_size, 5, 3, 256, 704).astype(np.float32),
        "intrinsics": np.tile(np.eye(3), (batch_size, 5, 1, 1)).astype(np.float32),
        "extrinsics": np.tile(np.eye(4), (batch_size, 5, 1, 1)).astype(np.float32),
        "points": np.random.randn(batch_size, 150000, 4).astype(np.float32),
        "points_mask": np.ones((batch_size, 150000), dtype=bool),
    }
    
    # Warmup
    logger.info("Warming up...")
    for _ in range(num_warmup):
        session.run(None, inputs)
    
    # Benchmark
    logger.info(f"Running {num_iterations} iterations...")
    latencies = []
    
    for _ in range(num_iterations):
        start = time.perf_counter()
        session.run(None, inputs)
        latencies.append((time.perf_counter() - start) * 1000)
    
    return {
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
        "p50_ms": np.percentile(latencies, 50),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "fps": 1000 / np.mean(latencies),
    }


def benchmark_coreml(
    model_path: str,
    batch_size: int = 1,
    num_warmup: int = 10,
    num_iterations: int = 100,
):
    """Benchmark CoreML model (Apple Silicon only)."""
    import coremltools as ct
    
    logger = logging.getLogger(__name__)
    
    model = ct.models.MLModel(model_path)
    
    # Create inputs
    inputs = {
        "images": np.random.randn(batch_size, 5, 3, 256, 704).astype(np.float32),
        "intrinsics": np.tile(np.eye(3), (batch_size, 5, 1, 1)).astype(np.float32),
        "extrinsics": np.tile(np.eye(4), (batch_size, 5, 1, 1)).astype(np.float32),
        "points": np.random.randn(batch_size, 150000, 4).astype(np.float32),
        "points_mask": np.ones((batch_size, 150000), dtype=np.float32),
    }
    
    # Warmup
    logger.info("Warming up...")
    for _ in range(num_warmup):
        model.predict(inputs)
    
    # Benchmark
    logger.info(f"Running {num_iterations} iterations...")
    latencies = []
    
    for _ in range(num_iterations):
        start = time.perf_counter()
        model.predict(inputs)
        latencies.append((time.perf_counter() - start) * 1000)
    
    return {
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
        "p50_ms": np.percentile(latencies, 50),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "fps": 1000 / np.mean(latencies),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark model inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to PyTorch checkpoint",
    )
    parser.add_argument(
        "--exported",
        type=str,
        default=None,
        help="Path to exported model (ONNX, CoreML)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pytorch",
        choices=["pytorch", "onnx", "coreml"],
        help="Model format",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "mps", "cpu"],
        help="Device for PyTorch inference",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 50)
    logger.info("Fusion2Drive Inference Benchmark")
    logger.info("=" * 50)
    
    if args.format == "pytorch":
        if args.checkpoint is None:
            logger.error("--checkpoint required for PyTorch benchmark")
            sys.exit(1)
        
        # Check device availability
        if args.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            args.device = "cpu"
        elif args.device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS not available, falling back to CPU")
            args.device = "cpu"
        
        logger.info(f"Device: {args.device}")
        if args.device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Load model
        logger.info(f"Loading model: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        
        model_config = checkpoint.get("config", {})
        if isinstance(model_config, dict):
            model_config = FusionModelConfig(**model_config)
        
        model = FusionModel(model_config)
        model.load_state_dict(checkpoint["model"])
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {num_params:,}")
        
        # Benchmark
        results = benchmark_pytorch(
            model, args.device, args.batch_size, args.num_warmup, args.num_iterations
        )
    
    elif args.format == "onnx":
        if args.exported is None:
            logger.error("--exported required for ONNX benchmark")
            sys.exit(1)
        
        logger.info(f"Loading ONNX model: {args.exported}")
        results = benchmark_onnx(
            args.exported, args.batch_size, args.num_warmup, args.num_iterations
        )
    
    elif args.format == "coreml":
        if args.exported is None:
            logger.error("--exported required for CoreML benchmark")
            sys.exit(1)
        
        logger.info(f"Loading CoreML model: {args.exported}")
        results = benchmark_coreml(
            args.exported, args.batch_size, args.num_warmup, args.num_iterations
        )
    
    # Print results
    logger.info("\n" + "=" * 50)
    logger.info("Results")
    logger.info("-" * 50)
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Format: {args.format}")
    logger.info("-" * 50)
    logger.info(f"Mean latency: {results['mean_ms']:.2f} ms")
    logger.info(f"Std latency:  {results['std_ms']:.2f} ms")
    logger.info(f"Min latency:  {results['min_ms']:.2f} ms")
    logger.info(f"Max latency:  {results['max_ms']:.2f} ms")
    logger.info(f"P50 latency:  {results['p50_ms']:.2f} ms")
    logger.info(f"P95 latency:  {results['p95_ms']:.2f} ms")
    logger.info(f"P99 latency:  {results['p99_ms']:.2f} ms")
    logger.info("-" * 50)
    logger.info(f"Throughput: {results['fps']:.1f} FPS")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
