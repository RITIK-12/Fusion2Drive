#!/usr/bin/env python3
"""
CARLA closed-loop simulation for Fusion2Drive.

Usage:
    python scripts/run_carla_sim.py --checkpoint checkpoints/best.pt --config configs/carla_sim.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wod_fusion.models import FusionModel, FusionModelConfig
from wod_fusion.sim import CarlaClient, CarlaConfig, SensorInterface, SensorConfig
from wod_fusion.sim.controllers import create_controller


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path: str, device: str) -> FusionModel:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    model_config = checkpoint.get("config", {})
    if isinstance(model_config, dict):
        model_config = FusionModelConfig(**model_config)
    
    model = FusionModel(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    
    return model


def run_simulation(
    model: FusionModel,
    config: dict,
    device: str,
    max_steps: int = 1000,
):
    """Run closed-loop simulation."""
    logger = logging.getLogger(__name__)
    
    # Create CARLA client
    carla_config = CarlaConfig(
        host=config["server"]["host"],
        port=config["server"]["port"],
        map_name=config["world"]["map_name"],
        weather=config["world"]["weather"],
        fixed_delta_seconds=config["world"]["fixed_delta_seconds"],
    )
    
    client = CarlaClient(carla_config)
    
    # Create sensor interface
    sensor_interface = SensorInterface()
    
    # Create controller
    controller_config = config["controller"]
    controller = create_controller(
        controller_config["type"],
        **controller_config.get(controller_config["type"], {}),
    )
    
    try:
        # Connect to CARLA
        if not client.connect():
            logger.error("Failed to connect to CARLA server")
            return
        
        # Spawn ego vehicle
        if not client.spawn_ego_vehicle():
            logger.error("Failed to spawn ego vehicle")
            return
        
        # Spawn sensors
        client.spawn_sensors()
        
        # Spawn traffic
        traffic_config = config.get("traffic", {})
        client.spawn_traffic(
            num_vehicles=traffic_config.get("num_vehicles", 50),
            num_pedestrians=traffic_config.get("num_pedestrians", 30),
        )
        
        # Let world settle
        for _ in range(20):
            client.tick()
        
        logger.info("Starting simulation loop...")
        
        # Simulation metrics
        metrics = {
            "distance_traveled": 0.0,
            "speeds": [],
            "collisions": 0,
            "inference_times": [],
        }
        
        last_position = None
        
        for step in range(max_steps):
            # Get sensor data
            sensor_data = client.get_sensor_data()
            
            # Prepare model input
            model_input = sensor_interface.prepare_model_input(sensor_data)
            
            # Convert to tensors
            images = torch.from_numpy(model_input["images"]).unsqueeze(0).to(device)
            intrinsics = torch.from_numpy(model_input["intrinsics"]).unsqueeze(0).to(device)
            extrinsics = torch.from_numpy(model_input["extrinsics"]).unsqueeze(0).to(device)
            points = torch.from_numpy(model_input["points"]).unsqueeze(0).to(device)
            points_mask = torch.from_numpy(model_input["points_mask"]).unsqueeze(0).to(device)
            
            # Run inference
            import time
            start_time = time.perf_counter()
            
            with torch.no_grad():
                outputs = model.inference(
                    images=images,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    points=points,
                    points_mask=points_mask,
                )
            
            inference_time = (time.perf_counter() - start_time) * 1000
            metrics["inference_times"].append(inference_time)
            
            # Get predicted waypoints
            waypoints = outputs["waypoints"][0].cpu().numpy()
            
            # Transform waypoints from Waymo to CARLA coords
            waypoints_carla = sensor_interface.waymo_to_carla_coords(waypoints)
            
            # Get ego state
            ego_state = client.get_ego_state()
            
            # Compute control
            throttle, steer, brake = controller.compute_control(
                current_position=ego_state["position"][:2],
                current_heading=ego_state["heading"],
                current_speed=ego_state["speed"],
                waypoints=waypoints_carla[:, :2],
            )
            
            # Apply control
            client.apply_control(throttle=throttle, steer=steer, brake=brake)
            
            # Step simulation
            client.tick()
            
            # Update metrics
            if last_position is not None:
                distance = np.linalg.norm(ego_state["position"][:2] - last_position)
                metrics["distance_traveled"] += distance
            
            last_position = ego_state["position"][:2].copy()
            metrics["speeds"].append(ego_state["speed"])
            
            # Log progress
            if step % 100 == 0:
                avg_speed = np.mean(metrics["speeds"][-100:]) if metrics["speeds"] else 0
                avg_inference = np.mean(metrics["inference_times"][-100:]) if metrics["inference_times"] else 0
                
                logger.info(
                    f"Step {step}/{max_steps} | "
                    f"Speed: {avg_speed:.1f} m/s | "
                    f"Distance: {metrics['distance_traveled']:.1f} m | "
                    f"Inference: {avg_inference:.1f} ms"
                )
        
        # Final metrics
        logger.info("\n" + "=" * 50)
        logger.info("Simulation Complete!")
        logger.info("-" * 50)
        logger.info(f"Total distance: {metrics['distance_traveled']:.1f} m")
        logger.info(f"Average speed: {np.mean(metrics['speeds']):.1f} m/s")
        logger.info(f"Average inference time: {np.mean(metrics['inference_times']):.1f} ms")
        logger.info("=" * 50)
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    
    finally:
        # Cleanup
        client.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Run CARLA simulation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/carla_sim.yaml",
        help="Path to CARLA simulation config",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=2000,
        help="Maximum simulation steps",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device to use (-1 for CPU)",
    )
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Load config
    config = load_config(args.config)
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    
    # Run simulation
    run_simulation(model, config, device, args.max_steps)


if __name__ == "__main__":
    main()
