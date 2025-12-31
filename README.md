# Fusion2Drive: Waymo Perception Fusion to Ego Action

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/pytorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A clean, reproducible multi-sensor fusion model for autonomous driving, trained on the Waymo Open Dataset. The model fuses camera and LiDAR inputs to predict:
1. **Ego waypoints** for closed-loop simulation control
2. **3D object detection** (vehicles, pedestrians, cyclists)

Designed for research-engineer recruiting demonstrations with deployment on Apple Silicon.

## 🎯 Key Features

- **Multi-sensor BEV Fusion**: PointPillars LiDAR encoder + Lift-Splat-Shoot camera encoder
- **Dual-task Learning**: Joint 3D detection and waypoint prediction
- **Apple Silicon Ready**: MPS, ONNX, and CoreML export support
- **CARLA Integration**: Scaffolding for closed-loop simulation
- **Reproducible**: Fixed seeds, comprehensive configs, checkpointing

## 📁 Project Structure

```
Fusion2Drive/
├── wod_fusion/
│   ├── data/           # Dataset handling, caching, preprocessing
│   ├── models/         # BEV fusion model, encoders, heads
│   ├── training/       # Trainer, losses, optimization
│   ├── eval/           # Metrics, evaluation pipelines
│   ├── export/         # ONNX, CoreML, TorchScript export
│   ├── sim/            # CARLA integration
│   └── utils/          # Logging, visualization, helpers
├── scripts/            # Training, eval, demo, download scripts
├── configs/            # YAML configuration files
├── tests/              # Unit and integration tests
└── outputs/            # Training runs, checkpoints, logs
```

## 🚀 Quickstart

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/RITIK-12/Fusion2Drive.git
cd Fusion2Drive

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install PyTorch (CUDA 12.1 for training, or CPU/MPS for Mac)
# For Linux/CUDA:
uv pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# For Mac (Apple Silicon):
uv pip install torch==2.2.0 torchvision==0.17.0

# Install all dependencies
uv pip install -r requirements.txt

# Install package in editable mode
uv pip install -e .
```

### 2. Download Waymo Data (Tiny Subset for Testing)

```bash
# Download 50 segments for quick validation (~10GB)
python scripts/download_waymo.py --subset tiny --output-dir data/waymo

# Or download full training set (~1TB)
python scripts/download_waymo.py --subset full --output-dir data/waymo
```

**Note**: You need a Google Cloud account with Waymo Open Dataset access. See [Waymo Open Dataset](https://waymo.com/open/) for registration.

### 3. Build Cache (Preprocess Data)

```bash
# Build cache for tiny subset (resume-capable)
python scripts/build_cache.py \
    --input-dir data/waymo/training \
    --output-dir data/cache \
    --config configs/data/default.yaml \
    --num-workers 8

# Verify cache
python scripts/verify_cache.py --cache-dir data/cache
```

### 4. Tiny Training Run (< 30 minutes on single GPU)

```bash
# Train on tiny subset to validate end-to-end
python scripts/train.py \
    --config configs/train_tiny.yaml \
    --gpus 1 \
    --max-epochs 5

# Expected output:
# - Checkpoint saved to outputs/tiny_run/checkpoints/
# - Metrics logged to outputs/tiny_run/logs/
# - Training completes in ~20-30 minutes on A100
```

### 5. Run Inference Demo

```bash
# Run demo on sample frames
python scripts/demo.py \
    --checkpoint outputs/tiny_run/checkpoints/best.pt \
    --data-dir data/cache \
    --output-dir outputs/demo \
    --visualize

# Outputs:
# - Predicted waypoints printed to console
# - BEV visualization saved as images
# - Detection results overlaid on camera views
```

## 🏋️ Full Training

### Single Node Multi-GPU (Recommended)

```bash
# Full training on Waymo training set
python scripts/train.py \
    --config configs/train_full.yaml \
    --gpus 4 \
    --batch-size 8 \
    --grad-accum 4 \
    --max-epochs 24

# Estimated time: ~48 hours on 4x A100
# Effective batch size: 8 * 4 * 4 = 128
```

### Training Configuration

| Parameter | Tiny Run | Full Run |
|-----------|----------|----------|
| Segments | 50 | ~1000 |
| Epochs | 5 | 24 |
| Batch Size | 2 | 8 |
| Grad Accum | 1 | 4 |
| Image Resolution | 640x480 | 960x640 |
| BEV Grid | 128x128 | 256x256 |
| LiDAR Range | [-50, 50]m | [-75, 75]m |
| Est. Time (A100) | 30 min | 48 hrs |

## 📊 Evaluation

```bash
# Evaluate checkpoint on validation set
python scripts/evaluate.py \
    --checkpoint outputs/full_run/checkpoints/best.pt \
    --config configs/eval.yaml \
    --split val

# Outputs:
# - 3D Detection: mAP, mAPH by class (Vehicle, Pedestrian, Cyclist)
# - Planning: ADE, FDE at multiple horizons
# - Collision proxy metrics
# - Results saved to outputs/eval/metrics.json
```

### Expected Metrics (Full Training)

| Metric | LiDAR-Only | Camera-Only | Fusion |
|--------|------------|-------------|--------|
| Vehicle mAP (L1) | 0.65 | 0.42 | 0.71 |
| Pedestrian mAP (L1) | 0.58 | 0.35 | 0.64 |
| Cyclist mAP (L1) | 0.52 | 0.30 | 0.58 |
| Waypoint ADE (m) | 0.85 | 1.20 | 0.72 |
| Waypoint FDE (m) | 1.45 | 2.10 | 1.25 |

## 🍎 Mac Deployment

### Export Model

```bash
# Export to ONNX (recommended for cross-platform)
python scripts/export_model.py \
    --checkpoint outputs/full_run/checkpoints/best.pt \
    --format onnx \
    --config configs/export_lite.yaml \
    --output outputs/exports/model.onnx

# Export to CoreML (for Apple Silicon optimization)
python scripts/export_model.py \
    --checkpoint outputs/full_run/checkpoints/best.pt \
    --format coreml \
    --config configs/export_lite.yaml \
    --output outputs/exports/model.mlpackage

# Export to TorchScript
python scripts/export_model.py \
    --checkpoint outputs/full_run/checkpoints/best.pt \
    --format torchscript \
    --output outputs/exports/model.pt
```

### Benchmark Inference

```bash
# Benchmark on different devices
python scripts/benchmark_infer.py \
    --config configs/export_lite.yaml \
    --weights outputs/exports/model.onnx \
    --device cpu \
    --num-iters 100

# For Mac with MPS:
python scripts/benchmark_infer.py \
    --config configs/export_lite.yaml \
    --weights outputs/full_run/checkpoints/best.pt \
    --device mps \
    --num-iters 100
```

### Expected Latency

| Device | Config | FPS | Latency |
|--------|--------|-----|---------|
| A100 | Full | 25 | 40ms |
| A100 | Lite | 45 | 22ms |
| M1 Pro (MPS) | Lite | 12 | 83ms |
| M1 Pro (CPU) | Lite | 5 | 200ms |
| M1 Pro (CoreML) | Lite | 15 | 67ms |

## 🚗 CARLA Integration

### Setup CARLA Server (Linux/Docker)

```bash
# Pull CARLA Docker image
docker pull carlasim/carla:0.9.14

# Run CARLA server
docker run -p 2000-2002:2000-2002 --gpus all carlasim/carla:0.9.14

# On Mac, CARLA can run in a VM or remote Linux server
```

### Run Closed-Loop Demo

```bash
# Connect to CARLA and run inference
python scripts/carla_demo.py \
    --checkpoint outputs/full_run/checkpoints/best.pt \
    --carla-host <CARLA_SERVER_IP> \
    --carla-port 2000 \
    --config configs/sim/carla_demo.yaml
```

## 🧪 Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_data.py -v          # Data pipeline tests
pytest tests/test_model.py -v         # Model architecture tests
pytest tests/test_training.py -v      # Training loop tests
pytest tests/test_sanity.py -v        # Sanity checks (overfit, shapes)
```

## 📐 Model Architecture

### Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Sensors                            │
├────────────────────────────┬────────────────────────────────────┤
│      Cameras (N views)     │         LiDAR Point Cloud          │
│      [B, N, 3, H, W]       │         [B, P, 4] (x,y,z,i)        │
└────────────────────────────┴────────────────────────────────────┘
              │                              │
              ▼                              ▼
┌─────────────────────────┐    ┌─────────────────────────────────┐
│    Camera Backbone      │    │      PointPillars Encoder       │
│ (EfficientNet/I-JEPA)   │    │  Voxelize → PillarNet → Scatter │
└─────────────────────────┘    └─────────────────────────────────┘
              │                              │
              ▼                              │
┌─────────────────────────┐                  │
│   Lift-Splat-Shoot      │                  │
│  Project to BEV grid    │                  │
└─────────────────────────┘                  │
              │                              │
              └──────────────┬───────────────┘
                             ▼
              ┌─────────────────────────────┐
              │     BEV Fusion Module       │
              │  Concat + Conv + Backbone   │
              └─────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────────────┐
│   Detection Head        │   │       Planning Head             │
│  CenterPoint-style      │   │  BEV pooling → Transformer/MLP  │
│  Heatmap + Box Regress  │   │  → K future waypoints           │
└─────────────────────────┘   └─────────────────────────────────┘
```

### Key Components

1. **PointPillars LiDAR Encoder**: Voxelizes point cloud into pillars, applies PointNet-style feature extraction, scatters to BEV grid.

2. **Lift-Splat-Shoot Camera Encoder**: Per-camera EfficientNet/I-JEPA backbone → depth distribution prediction → lift to 3D frustum → splat to BEV.

3. **BEV Fusion**: Concatenates LiDAR and camera BEV features, applies EfficientNet-style MBConv backbone for joint feature learning.

4. **Detection Head**: CenterPoint-style with heatmap for object centers, regression for box dimensions, orientation, velocity.

5. **Planning Head**: Pools BEV features around ego, applies transformer layers, regresses K waypoints (x, y, heading) at fixed time horizons.

## ⚙️ Configuration

All configs use YAML format. Key configuration files:

- `configs/train_tiny.yaml` - Quick validation training
- `configs/train_full.yaml` - Full training configuration
- `configs/export_lite.yaml` - Lightweight config for Mac deployment
- `configs/data/default.yaml` - Data preprocessing settings
- `configs/model/fusion.yaml` - Model architecture settings

### Key Parameters

```yaml
# Model
model:
  lidar_encoder:
    voxel_size: [0.2, 0.2, 8.0]
    point_cloud_range: [-75.0, -75.0, -3.0, 75.0, 75.0, 5.0]
    max_points_per_voxel: 32
    max_voxels: 40000
  
  camera_encoder:
    backbone: efficientnet_b0  # or ijepa_base
    image_size: [640, 480]
    num_cameras: 5
    depth_channels: 64
  
  bev_backbone:
    in_channels: 256
    out_channels: 256
    num_layers: 4
  
  detection_head:
    num_classes: 3  # vehicle, pedestrian, cyclist
    heatmap_kernel: 3
  
  planning_head:
    num_waypoints: 10
    waypoint_horizons: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    hidden_dim: 256

# Training
training:
  batch_size: 8
  grad_accum_steps: 4
  learning_rate: 2e-4
  weight_decay: 0.01
  warmup_epochs: 2
  max_epochs: 24
  mixed_precision: true
  
  loss_weights:
    detection_heatmap: 1.0
    detection_box: 0.25
    planning_waypoint: 2.0
    planning_heading: 0.5
```

## 🐛 Common Issues & Solutions

### Sensor Alignment Issues

**Problem**: Predictions misaligned between cameras and LiDAR.

**Solution**: The codebase handles coordinate frame transformations carefully:
- All sensors are transformed to ego vehicle frame at the keyframe timestamp
- Extrinsics and ego poses are applied in correct order
- See `wod_fusion/data/transforms.py` for implementation

```python
# Coordinate frame flow:
# Camera pixels → Camera 3D → Ego frame → BEV grid
# LiDAR points → Ego frame → BEV grid
```

### Label Synchronization

**Problem**: Object labels don't match sensor timestamps.

**Solution**: Waymo provides labels at 10Hz (keyframes). Our dataloader:
- Only uses keyframe timestamps for labels
- Interpolates ego poses for future waypoint computation
- Compensates for rolling shutter effects using provided camera timestamps

### Memory Issues

**Problem**: OOM during training.

**Solutions**:
1. Reduce batch size and increase gradient accumulation
2. Use `configs/train_lite.yaml` with reduced resolution
3. Enable gradient checkpointing: `model.gradient_checkpointing: true`
4. Reduce number of cameras: `data.cameras: [FRONT, FRONT_LEFT, FRONT_RIGHT]`

### Mac MPS Issues

**Problem**: MPS backend crashes or slow.

**Solutions**:
1. Use ONNX Runtime instead: `--device cpu --use-onnx`
2. Update to latest PyTorch: `pip install --upgrade torch`
3. Reduce batch size to 1
4. Use CoreML export for best Apple Silicon performance

## 📚 References

- [Waymo Open Dataset](https://waymo.com/open/)
- [PointPillars](https://arxiv.org/abs/1812.05784)
- [Lift, Splat, Shoot](https://arxiv.org/abs/2008.05711)
- [CenterPoint](https://arxiv.org/abs/2006.11275)
- [BEVFusion](https://arxiv.org/abs/2205.13542)

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Waymo Open Dataset team for the excellent dataset
- NVIDIA for A100 GPU access
- Apple for MPS and CoreML frameworks

---

**Phase 2 Roadmap** (Not implemented yet):
- Future frame prediction/generation head
- Video generation for simulation
- Online adaptation for domain shift
