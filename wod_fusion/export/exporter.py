"""
Model export utilities for ONNX, TorchScript, and CoreML.
Optimized for Mac (Apple Silicon) inference via MPS/CoreML.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for model export."""
    
    # Model settings
    batch_size: int = 1
    num_cameras: int = 5
    image_size: Tuple[int, int] = (256, 704)  # H, W
    max_points: int = 150000
    num_waypoints: int = 12
    
    # BEV grid
    bev_size: Tuple[int, int] = (200, 200)  # H, W
    
    # Export settings
    opset_version: int = 17  # ONNX opset
    optimize_for_mobile: bool = False
    
    # Quantization
    quantize: bool = False
    quantization_backend: str = "qnnpack"  # or "fbgemm" for x86


class ExportableModel(nn.Module):
    """
    Wrapper for FusionModel that simplifies the interface for export.
    
    This removes complex control flow and makes the model more export-friendly.
    """
    
    def __init__(self, model: nn.Module, mode: str = "fusion"):
        """
        Args:
            model: The FusionModel to export
            mode: One of "fusion", "lidar_only", "camera_only"
        """
        super().__init__()
        self.mode = mode
        
        # Copy relevant components
        if mode in ["fusion", "lidar_only"]:
            self.lidar_encoder = model.lidar_encoder
        else:
            self.lidar_encoder = None
        
        if mode in ["fusion", "camera_only"]:
            self.camera_encoder = model.camera_encoder
        else:
            self.camera_encoder = None
        
        if mode == "fusion":
            self.fusion_conv = model.fusion_conv
        
        self.bev_backbone = model.bev_backbone
        
        # Use simpler heads for export
        if hasattr(model, "planning_head_lite") and model.planning_head_lite is not None:
            self.planning_head = model.planning_head_lite
        else:
            self.planning_head = model.planning_head
        
        self.detection_head = model.detection_head
    
    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        points: torch.Tensor,
        points_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for export.
        
        Args:
            images: (B, N, 3, H, W)
            intrinsics: (B, N, 3, 3)
            extrinsics: (B, N, 4, 4)
            points: (B, P, 4)
            points_mask: (B, P)
            
        Returns:
            Dict with 'heatmap', 'box_reg', 'waypoints'
        """
        batch_size = images.shape[0]
        
        # Encode modalities
        if self.mode == "lidar_only":
            bev_features = self.lidar_encoder(points, points_mask)
        elif self.mode == "camera_only":
            bev_features = self.camera_encoder(images, intrinsics, extrinsics)
        else:
            lidar_bev = self.lidar_encoder(points, points_mask)
            camera_bev = self.camera_encoder(images, intrinsics, extrinsics)
            bev_features = self.fusion_conv(torch.cat([lidar_bev, camera_bev], dim=1))
        
        # BEV backbone
        bev_features = self.bev_backbone(bev_features)
        
        # Detection head
        det_output = self.detection_head(bev_features)
        
        # Planning head
        waypoints = self.planning_head(bev_features)
        
        return {
            "heatmap": det_output["heatmap"],
            "box_reg": det_output["box_reg"],
            "waypoints": waypoints,
        }


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    config: Optional[ExportConfig] = None,
    mode: str = "fusion",
    verify: bool = True,
) -> str:
    """
    Export model to ONNX format.
    
    Args:
        model: FusionModel to export
        output_path: Path for output file
        config: Export configuration
        mode: One of "fusion", "lidar_only", "camera_only"
        verify: Whether to verify the exported model
        
    Returns:
        Path to exported model
    """
    if config is None:
        config = ExportConfig()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create exportable wrapper
    export_model = ExportableModel(model, mode=mode)
    export_model.eval()
    
    # Create dummy inputs
    B = config.batch_size
    N = config.num_cameras
    H, W = config.image_size
    P = config.max_points
    
    device = next(model.parameters()).device
    
    dummy_images = torch.randn(B, N, 3, H, W, device=device)
    dummy_intrinsics = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
    dummy_extrinsics = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
    dummy_points = torch.randn(B, P, 4, device=device)
    dummy_mask = torch.ones(B, P, dtype=torch.bool, device=device)
    
    dummy_inputs = (
        dummy_images,
        dummy_intrinsics,
        dummy_extrinsics,
        dummy_points,
        dummy_mask,
    )
    
    input_names = ["images", "intrinsics", "extrinsics", "points", "points_mask"]
    output_names = ["heatmap", "box_reg", "waypoints"]
    
    # Dynamic axes for variable batch size
    dynamic_axes = {
        "images": {0: "batch_size"},
        "intrinsics": {0: "batch_size"},
        "extrinsics": {0: "batch_size"},
        "points": {0: "batch_size"},
        "points_mask": {0: "batch_size"},
        "heatmap": {0: "batch_size"},
        "box_reg": {0: "batch_size"},
        "waypoints": {0: "batch_size"},
    }
    
    logger.info(f"Exporting model to ONNX: {output_path}")
    
    torch.onnx.export(
        export_model,
        dummy_inputs,
        str(output_path),
        opset_version=config.opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    
    # Optimize ONNX model
    try:
        import onnx
        from onnxsim import simplify
        
        logger.info("Simplifying ONNX model...")
        onnx_model = onnx.load(str(output_path))
        onnx_model_simp, check = simplify(onnx_model)
        
        if check:
            onnx.save(onnx_model_simp, str(output_path))
            logger.info("Successfully simplified ONNX model")
        else:
            logger.warning("ONNX simplification check failed, using original")
    except ImportError:
        logger.warning("onnx-simplifier not installed, skipping optimization")
    
    # Verify
    if verify:
        try:
            import onnx
            import onnxruntime as ort
            
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model validation passed")
            
            # Test inference
            session = ort.InferenceSession(str(output_path))
            
            inputs = {
                "images": dummy_images.cpu().numpy(),
                "intrinsics": dummy_intrinsics.cpu().numpy(),
                "extrinsics": dummy_extrinsics.cpu().numpy(),
                "points": dummy_points.cpu().numpy(),
                "points_mask": dummy_mask.cpu().numpy(),
            }
            
            outputs = session.run(None, inputs)
            logger.info(f"ONNX inference test passed, output shapes: {[o.shape for o in outputs]}")
        except ImportError:
            logger.warning("onnxruntime not installed, skipping verification")
    
    logger.info(f"Successfully exported to {output_path}")
    return str(output_path)


def export_to_torchscript(
    model: nn.Module,
    output_path: str,
    config: Optional[ExportConfig] = None,
    mode: str = "fusion",
    method: str = "trace",
    optimize_for_mobile: bool = False,
) -> str:
    """
    Export model to TorchScript format.
    
    Args:
        model: FusionModel to export
        output_path: Path for output file
        config: Export configuration
        mode: One of "fusion", "lidar_only", "camera_only"
        method: "trace" or "script"
        optimize_for_mobile: Whether to optimize for mobile deployment
        
    Returns:
        Path to exported model
    """
    if config is None:
        config = ExportConfig()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create exportable wrapper
    export_model = ExportableModel(model, mode=mode)
    export_model.eval()
    export_model.cpu()  # Move to CPU for export
    
    # Create dummy inputs
    B = config.batch_size
    N = config.num_cameras
    H, W = config.image_size
    P = config.max_points
    
    dummy_images = torch.randn(B, N, 3, H, W)
    dummy_intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
    dummy_extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
    dummy_points = torch.randn(B, P, 4)
    dummy_mask = torch.ones(B, P, dtype=torch.bool)
    
    logger.info(f"Exporting model to TorchScript ({method}): {output_path}")
    
    if method == "trace":
        traced = torch.jit.trace(
            export_model,
            (dummy_images, dummy_intrinsics, dummy_extrinsics, dummy_points, dummy_mask),
        )
        scripted = traced
    else:
        scripted = torch.jit.script(export_model)
    
    # Optimize for mobile if requested
    if optimize_for_mobile:
        try:
            from torch.utils.mobile_optimizer import optimize_for_mobile
            
            logger.info("Optimizing for mobile...")
            scripted = optimize_for_mobile(scripted)
        except ImportError:
            logger.warning("Mobile optimizer not available")
    
    # Save
    scripted.save(str(output_path))
    
    # Verify
    loaded = torch.jit.load(str(output_path))
    output = loaded(dummy_images, dummy_intrinsics, dummy_extrinsics, dummy_points, dummy_mask)
    logger.info(f"TorchScript verification passed, waypoints shape: {output['waypoints'].shape}")
    
    logger.info(f"Successfully exported to {output_path}")
    return str(output_path)


def export_to_coreml(
    model: nn.Module,
    output_path: str,
    config: Optional[ExportConfig] = None,
    mode: str = "fusion",
    convert_to_mlpackage: bool = True,
) -> str:
    """
    Export model to CoreML format for Apple devices.
    
    Args:
        model: FusionModel to export
        output_path: Path for output file
        config: Export configuration
        mode: One of "fusion", "lidar_only", "camera_only"
        convert_to_mlpackage: Use .mlpackage format (recommended for new models)
        
    Returns:
        Path to exported model
    """
    try:
        import coremltools as ct
    except ImportError:
        raise ImportError("coremltools is required for CoreML export: pip install coremltools")
    
    if config is None:
        config = ExportConfig()
    
    output_path = Path(output_path)
    if convert_to_mlpackage and not str(output_path).endswith(".mlpackage"):
        output_path = output_path.with_suffix(".mlpackage")
    elif not convert_to_mlpackage and not str(output_path).endswith(".mlmodel"):
        output_path = output_path.with_suffix(".mlmodel")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # First export to TorchScript
    logger.info("First exporting to TorchScript...")
    ts_path = output_path.with_suffix(".pt")
    
    export_model = ExportableModel(model, mode=mode)
    export_model.eval()
    export_model.cpu()
    
    B = config.batch_size
    N = config.num_cameras
    H, W = config.image_size
    P = config.max_points
    
    dummy_images = torch.randn(B, N, 3, H, W)
    dummy_intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).contiguous()
    dummy_extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).contiguous()
    dummy_points = torch.randn(B, P, 4)
    dummy_mask = torch.ones(B, P, dtype=torch.bool)
    
    traced = torch.jit.trace(
        export_model,
        (dummy_images, dummy_intrinsics, dummy_extrinsics, dummy_points, dummy_mask),
    )
    
    # Convert to CoreML
    logger.info(f"Converting to CoreML: {output_path}")
    
    # Define input types
    image_input = ct.TensorType(name="images", shape=(B, N, 3, H, W))
    intrinsics_input = ct.TensorType(name="intrinsics", shape=(B, N, 3, 3))
    extrinsics_input = ct.TensorType(name="extrinsics", shape=(B, N, 4, 4))
    points_input = ct.TensorType(name="points", shape=(B, P, 4))
    mask_input = ct.TensorType(name="points_mask", shape=(B, P))
    
    mlmodel = ct.convert(
        traced,
        inputs=[image_input, intrinsics_input, extrinsics_input, points_input, mask_input],
        convert_to="mlprogram" if convert_to_mlpackage else "neuralnetwork",
        compute_precision=ct.precision.FLOAT16 if config.optimize_for_mobile else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS16 if convert_to_mlpackage else None,
    )
    
    # Set metadata
    mlmodel.author = "Fusion2Drive"
    mlmodel.short_description = "Multi-sensor fusion model for autonomous driving"
    mlmodel.version = "1.0.0"
    
    # Save
    mlmodel.save(str(output_path))
    
    # Clean up temp files
    if ts_path.exists():
        ts_path.unlink()
    
    logger.info(f"Successfully exported to {output_path}")
    return str(output_path)


class ModelExporter:
    """
    High-level interface for exporting models to multiple formats.
    """
    
    def __init__(
        self,
        model: nn.Module,
        output_dir: str = "exports",
        config: Optional[ExportConfig] = None,
    ):
        """
        Args:
            model: FusionModel to export
            output_dir: Base directory for exports
            config: Export configuration
        """
        self.model = model
        self.model.eval()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or ExportConfig()
    
    def export_all(
        self,
        mode: str = "fusion",
        formats: Optional[list] = None,
    ) -> Dict[str, str]:
        """
        Export model to all supported formats.
        
        Args:
            mode: One of "fusion", "lidar_only", "camera_only"
            formats: List of formats to export (default: ["onnx", "torchscript", "coreml"])
            
        Returns:
            Dict mapping format to output path
        """
        if formats is None:
            formats = ["onnx", "torchscript", "coreml"]
        
        results = {}
        
        if "onnx" in formats:
            try:
                path = export_to_onnx(
                    self.model,
                    self.output_dir / f"model_{mode}.onnx",
                    self.config,
                    mode=mode,
                )
                results["onnx"] = path
            except Exception as e:
                logger.error(f"ONNX export failed: {e}")
        
        if "torchscript" in formats:
            try:
                path = export_to_torchscript(
                    self.model,
                    self.output_dir / f"model_{mode}.pt",
                    self.config,
                    mode=mode,
                )
                results["torchscript"] = path
            except Exception as e:
                logger.error(f"TorchScript export failed: {e}")
        
        if "coreml" in formats:
            try:
                path = export_to_coreml(
                    self.model,
                    self.output_dir / f"model_{mode}.mlpackage",
                    self.config,
                    mode=mode,
                )
                results["coreml"] = path
            except Exception as e:
                logger.error(f"CoreML export failed: {e}")
        
        return results
    
    def benchmark(
        self,
        format: str,
        model_path: str,
        num_warmup: int = 10,
        num_iterations: int = 100,
    ) -> Dict[str, float]:
        """
        Benchmark exported model inference speed.
        
        Args:
            format: One of "onnx", "torchscript", "coreml"
            model_path: Path to exported model
            num_warmup: Number of warmup iterations
            num_iterations: Number of timed iterations
            
        Returns:
            Dict with latency stats
        """
        import time
        
        B = self.config.batch_size
        N = self.config.num_cameras
        H, W = self.config.image_size
        P = self.config.max_points
        
        latencies = []
        
        if format == "onnx":
            import onnxruntime as ort
            
            session = ort.InferenceSession(model_path)
            
            inputs = {
                "images": torch.randn(B, N, 3, H, W).numpy(),
                "intrinsics": torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).numpy(),
                "extrinsics": torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).numpy(),
                "points": torch.randn(B, P, 4).numpy(),
                "points_mask": torch.ones(B, P, dtype=torch.bool).numpy(),
            }
            
            # Warmup
            for _ in range(num_warmup):
                session.run(None, inputs)
            
            # Benchmark
            for _ in range(num_iterations):
                start = time.perf_counter()
                session.run(None, inputs)
                latencies.append((time.perf_counter() - start) * 1000)
        
        elif format == "torchscript":
            model = torch.jit.load(model_path)
            model.eval()
            
            dummy_images = torch.randn(B, N, 3, H, W)
            dummy_intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
            dummy_extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
            dummy_points = torch.randn(B, P, 4)
            dummy_mask = torch.ones(B, P, dtype=torch.bool)
            
            # Warmup
            for _ in range(num_warmup):
                model(dummy_images, dummy_intrinsics, dummy_extrinsics, dummy_points, dummy_mask)
            
            # Benchmark
            for _ in range(num_iterations):
                start = time.perf_counter()
                model(dummy_images, dummy_intrinsics, dummy_extrinsics, dummy_points, dummy_mask)
                latencies.append((time.perf_counter() - start) * 1000)
        
        elif format == "coreml":
            import coremltools as ct
            
            model = ct.models.MLModel(model_path)
            
            inputs = {
                "images": torch.randn(B, N, 3, H, W).numpy(),
                "intrinsics": torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).numpy(),
                "extrinsics": torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).numpy(),
                "points": torch.randn(B, P, 4).numpy(),
                "points_mask": torch.ones(B, P, dtype=torch.float32).numpy(),
            }
            
            # Warmup
            for _ in range(num_warmup):
                model.predict(inputs)
            
            # Benchmark
            for _ in range(num_iterations):
                start = time.perf_counter()
                model.predict(inputs)
                latencies.append((time.perf_counter() - start) * 1000)
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        import numpy as np
        
        return {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "fps": float(1000 / np.mean(latencies)),
        }


def quantize_onnx(
    model_path: str,
    output_path: Optional[str] = None,
    quantization_type: str = "dynamic",
) -> str:
    """
    Quantize ONNX model for faster inference.
    
    Args:
        model_path: Path to input ONNX model
        output_path: Path for quantized model (default: add _quantized suffix)
        quantization_type: "dynamic" or "static"
        
    Returns:
        Path to quantized model
    """
    from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
    
    model_path = Path(model_path)
    
    if output_path is None:
        output_path = model_path.parent / f"{model_path.stem}_quantized{model_path.suffix}"
    else:
        output_path = Path(output_path)
    
    logger.info(f"Quantizing ONNX model ({quantization_type}): {model_path} -> {output_path}")
    
    if quantization_type == "dynamic":
        quantize_dynamic(
            str(model_path),
            str(output_path),
            weight_type=QuantType.QUInt8,
        )
    else:
        raise NotImplementedError("Static quantization requires calibration data")
    
    # Report size reduction
    original_size = model_path.stat().st_size / (1024 * 1024)  # MB
    quantized_size = output_path.stat().st_size / (1024 * 1024)  # MB
    
    logger.info(f"Model size: {original_size:.2f} MB -> {quantized_size:.2f} MB")
    logger.info(f"Size reduction: {(1 - quantized_size/original_size) * 100:.1f}%")
    
    return str(output_path)
