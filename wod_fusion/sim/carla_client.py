"""
CARLA client for closed-loop simulation.

This module provides the interface to connect to CARLA server,
spawn vehicles, and run simulation episodes.
"""

import logging
import time
import queue
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CarlaConfig:
    """Configuration for CARLA simulation."""
    
    # Server settings
    host: str = "localhost"
    port: int = 2000
    timeout: float = 30.0
    
    # World settings
    map_name: str = "Town03"
    weather: str = "ClearNoon"
    fixed_delta_seconds: float = 0.05  # 20 Hz
    
    # Ego vehicle settings
    vehicle_blueprint: str = "vehicle.tesla.model3"
    spawn_point_index: Optional[int] = None  # Random if None
    
    # Sensor setup (matching Waymo)
    num_cameras: int = 5
    camera_width: int = 1920
    camera_height: int = 1280
    camera_fov: int = 70
    
    # LiDAR settings
    lidar_channels: int = 64
    lidar_range: float = 75.0
    lidar_points_per_second: int = 2200000
    lidar_rotation_frequency: float = 20.0
    lidar_upper_fov: float = 3.0
    lidar_lower_fov: float = -25.0
    
    # Traffic settings
    num_vehicles: int = 50
    num_pedestrians: int = 30
    
    # Recording settings
    record: bool = False
    record_path: str = "recordings"
    
    # Rendering
    render: bool = True
    spectator_follow: bool = True


class CarlaClient:
    """
    Client for CARLA simulation.
    
    This class manages the connection to CARLA server, world setup,
    sensor spawning, and simulation stepping.
    """
    
    # Camera positions relative to ego vehicle (matching Waymo layout)
    CAMERA_TRANSFORMS = {
        "FRONT": {"location": (2.0, 0.0, 1.5), "rotation": (0, 0, 0)},
        "FRONT_LEFT": {"location": (1.5, -0.8, 1.5), "rotation": (0, -45, 0)},
        "FRONT_RIGHT": {"location": (1.5, 0.8, 1.5), "rotation": (0, 45, 0)},
        "SIDE_LEFT": {"location": (0.0, -1.0, 1.5), "rotation": (0, -90, 0)},
        "SIDE_RIGHT": {"location": (0.0, 1.0, 1.5), "rotation": (0, 90, 0)},
    }
    
    def __init__(self, config: Optional[CarlaConfig] = None):
        """
        Initialize CARLA client.
        
        Args:
            config: CARLA configuration
        """
        self.config = config or CarlaConfig()
        
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self.sensors = {}
        self.sensor_queues = {}
        self.traffic_manager = None
        self.spawned_actors = []
        
        self._connected = False
    
    def connect(self) -> bool:
        """
        Connect to CARLA server.
        
        Returns:
            True if connection successful
        """
        try:
            import carla
        except ImportError:
            logger.error(
                "CARLA Python API not installed. "
                "Please install from CARLA distribution or pip install carla"
            )
            return False
        
        try:
            logger.info(f"Connecting to CARLA at {self.config.host}:{self.config.port}")
            
            self.client = carla.Client(self.config.host, self.config.port)
            self.client.set_timeout(self.config.timeout)
            
            # Load world
            if self.config.map_name:
                logger.info(f"Loading map: {self.config.map_name}")
                self.world = self.client.load_world(self.config.map_name)
            else:
                self.world = self.client.get_world()
            
            # Set synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.config.fixed_delta_seconds
            self.world.apply_settings(settings)
            
            # Set weather
            self._set_weather(self.config.weather)
            
            # Setup traffic manager
            self.traffic_manager = self.client.get_trafficmanager()
            self.traffic_manager.set_synchronous_mode(True)
            
            self._connected = True
            logger.info("Successfully connected to CARLA")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to CARLA: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from CARLA and cleanup."""
        self.cleanup()
        
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        
        self._connected = False
        logger.info("Disconnected from CARLA")
    
    def spawn_ego_vehicle(self) -> bool:
        """
        Spawn the ego vehicle.
        
        Returns:
            True if successful
        """
        import carla
        
        if not self._connected:
            logger.error("Not connected to CARLA")
            return False
        
        blueprint_library = self.world.get_blueprint_library()
        
        # Get vehicle blueprint
        vehicle_bp = blueprint_library.find(self.config.vehicle_blueprint)
        if vehicle_bp.has_attribute("color"):
            vehicle_bp.set_attribute("color", "255,0,0")  # Red ego vehicle
        
        # Get spawn points
        spawn_points = self.world.get_map().get_spawn_points()
        
        if self.config.spawn_point_index is not None:
            spawn_point = spawn_points[self.config.spawn_point_index % len(spawn_points)]
        else:
            spawn_point = np.random.choice(spawn_points)
        
        # Spawn vehicle
        try:
            self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            self.spawned_actors.append(self.ego_vehicle)
            logger.info(f"Spawned ego vehicle: {self.ego_vehicle.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to spawn ego vehicle: {e}")
            return False
    
    def spawn_sensors(self) -> Dict[str, Any]:
        """
        Spawn sensors on ego vehicle.
        
        Returns:
            Dict mapping sensor name to actor
        """
        import carla
        
        if self.ego_vehicle is None:
            logger.error("Ego vehicle not spawned")
            return {}
        
        blueprint_library = self.world.get_blueprint_library()
        
        # Spawn cameras
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.config.camera_width))
        camera_bp.set_attribute("image_size_y", str(self.config.camera_height))
        camera_bp.set_attribute("fov", str(self.config.camera_fov))
        
        for name, transform_dict in self.CAMERA_TRANSFORMS.items():
            loc = transform_dict["location"]
            rot = transform_dict["rotation"]
            
            transform = carla.Transform(
                carla.Location(x=loc[0], y=loc[1], z=loc[2]),
                carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2])
            )
            
            camera = self.world.spawn_actor(
                camera_bp,
                transform,
                attach_to=self.ego_vehicle
            )
            
            # Setup data queue
            q = queue.Queue()
            camera.listen(lambda data, q=q: q.put(data))
            
            self.sensors[f"camera_{name}"] = camera
            self.sensor_queues[f"camera_{name}"] = q
            self.spawned_actors.append(camera)
            
            logger.info(f"Spawned camera: {name}")
        
        # Spawn LiDAR
        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", str(self.config.lidar_channels))
        lidar_bp.set_attribute("range", str(self.config.lidar_range))
        lidar_bp.set_attribute("points_per_second", str(self.config.lidar_points_per_second))
        lidar_bp.set_attribute("rotation_frequency", str(self.config.lidar_rotation_frequency))
        lidar_bp.set_attribute("upper_fov", str(self.config.lidar_upper_fov))
        lidar_bp.set_attribute("lower_fov", str(self.config.lidar_lower_fov))
        
        lidar_transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=2.4)  # On roof
        )
        
        lidar = self.world.spawn_actor(
            lidar_bp,
            lidar_transform,
            attach_to=self.ego_vehicle
        )
        
        q = queue.Queue()
        lidar.listen(lambda data, q=q: q.put(data))
        
        self.sensors["lidar"] = lidar
        self.sensor_queues["lidar"] = q
        self.spawned_actors.append(lidar)
        
        logger.info("Spawned LiDAR sensor")
        
        return self.sensors
    
    def spawn_traffic(self, num_vehicles: Optional[int] = None, num_pedestrians: Optional[int] = None):
        """
        Spawn traffic vehicles and pedestrians.
        
        Args:
            num_vehicles: Number of vehicles (default from config)
            num_pedestrians: Number of pedestrians (default from config)
        """
        import carla
        
        num_vehicles = num_vehicles or self.config.num_vehicles
        num_pedestrians = num_pedestrians or self.config.num_pedestrians
        
        blueprint_library = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()
        
        # Spawn vehicles
        vehicle_bps = blueprint_library.filter("vehicle.*")
        vehicle_bps = [bp for bp in vehicle_bps if int(bp.get_attribute("number_of_wheels")) == 4]
        
        spawn_points = spawn_points[:num_vehicles]
        
        for i, spawn_point in enumerate(spawn_points):
            bp = np.random.choice(vehicle_bps)
            
            try:
                vehicle = self.world.spawn_actor(bp, spawn_point)
                vehicle.set_autopilot(True, self.traffic_manager.get_port())
                self.spawned_actors.append(vehicle)
            except Exception:
                pass  # Skip if spawn fails
        
        logger.info(f"Spawned {len(spawn_points)} traffic vehicles")
        
        # Spawn pedestrians
        walker_bps = blueprint_library.filter("walker.pedestrian.*")
        spawn_locs = []
        
        for _ in range(num_pedestrians):
            loc = self.world.get_random_location_from_navigation()
            if loc:
                spawn_locs.append(carla.Transform(loc))
        
        for spawn_loc in spawn_locs:
            bp = np.random.choice(walker_bps)
            
            try:
                walker = self.world.spawn_actor(bp, spawn_loc)
                self.spawned_actors.append(walker)
            except Exception:
                pass
        
        logger.info(f"Spawned {len(spawn_locs)} pedestrians")
    
    def get_sensor_data(self, timeout: float = 10.0) -> Dict[str, np.ndarray]:
        """
        Get current sensor data.
        
        Args:
            timeout: Timeout for waiting on sensor data
            
        Returns:
            Dict mapping sensor name to numpy array
        """
        import carla
        
        data = {}
        
        for name, q in self.sensor_queues.items():
            try:
                sensor_data = q.get(timeout=timeout)
                
                if "camera" in name:
                    # Convert CARLA image to numpy
                    array = np.frombuffer(sensor_data.raw_data, dtype=np.uint8)
                    array = array.reshape((self.config.camera_height, self.config.camera_width, 4))
                    array = array[:, :, :3]  # Remove alpha
                    data[name] = array
                    
                elif name == "lidar":
                    # Convert CARLA point cloud to numpy
                    points = np.frombuffer(sensor_data.raw_data, dtype=np.float32)
                    points = points.reshape(-1, 4)  # x, y, z, intensity
                    data[name] = points
                    
            except queue.Empty:
                logger.warning(f"Timeout waiting for sensor: {name}")
                data[name] = None
        
        return data
    
    def get_ego_state(self) -> Dict[str, np.ndarray]:
        """
        Get current ego vehicle state.
        
        Returns:
            Dict with position, velocity, acceleration, heading
        """
        if self.ego_vehicle is None:
            return {}
        
        transform = self.ego_vehicle.get_transform()
        velocity = self.ego_vehicle.get_velocity()
        acceleration = self.ego_vehicle.get_acceleration()
        
        position = np.array([
            transform.location.x,
            transform.location.y,
            transform.location.z
        ])
        
        # Convert to radians
        heading = np.radians(transform.rotation.yaw)
        
        vel = np.array([velocity.x, velocity.y, velocity.z])
        acc = np.array([acceleration.x, acceleration.y, acceleration.z])
        
        return {
            "position": position,
            "heading": heading,
            "velocity": vel,
            "acceleration": acc,
            "speed": np.linalg.norm(vel[:2]),
        }
    
    def apply_control(
        self,
        throttle: float = 0.0,
        steer: float = 0.0,
        brake: float = 0.0,
        hand_brake: bool = False,
        reverse: bool = False,
    ):
        """
        Apply control to ego vehicle.
        
        Args:
            throttle: 0.0 to 1.0
            steer: -1.0 to 1.0
            brake: 0.0 to 1.0
            hand_brake: Hand brake engaged
            reverse: Reverse gear
        """
        import carla
        
        if self.ego_vehicle is None:
            return
        
        control = carla.VehicleControl(
            throttle=float(np.clip(throttle, 0.0, 1.0)),
            steer=float(np.clip(steer, -1.0, 1.0)),
            brake=float(np.clip(brake, 0.0, 1.0)),
            hand_brake=hand_brake,
            reverse=reverse,
        )
        
        self.ego_vehicle.apply_control(control)
    
    def tick(self) -> int:
        """
        Step simulation forward.
        
        Returns:
            Current frame number
        """
        if self.world:
            return self.world.tick()
        return 0
    
    def cleanup(self):
        """Cleanup spawned actors."""
        logger.info("Cleaning up spawned actors...")
        
        for actor in reversed(self.spawned_actors):
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception:
                pass
        
        self.spawned_actors = []
        self.sensors = {}
        self.sensor_queues = {}
        self.ego_vehicle = None
    
    def _set_weather(self, weather_preset: str):
        """Set weather preset."""
        import carla
        
        weather_presets = {
            "ClearNoon": carla.WeatherParameters.ClearNoon,
            "CloudyNoon": carla.WeatherParameters.CloudyNoon,
            "WetNoon": carla.WeatherParameters.WetNoon,
            "HardRainNoon": carla.WeatherParameters.HardRainNoon,
            "ClearSunset": carla.WeatherParameters.ClearSunset,
            "CloudySunset": carla.WeatherParameters.CloudySunset,
            "WetSunset": carla.WeatherParameters.WetSunset,
            "ClearNight": carla.WeatherParameters.ClearNight,
        }
        
        if weather_preset in weather_presets:
            self.world.set_weather(weather_presets[weather_preset])
            logger.info(f"Set weather: {weather_preset}")
        else:
            logger.warning(f"Unknown weather preset: {weather_preset}")
    
    def get_ground_truth(self) -> Dict[str, Any]:
        """
        Get ground truth for current frame.
        
        Returns:
            Dict with actor positions, velocities, bounding boxes
        """
        import carla
        
        if not self._connected:
            return {}
        
        gt = {
            "vehicles": [],
            "pedestrians": [],
            "ego_state": self.get_ego_state(),
        }
        
        for actor in self.world.get_actors():
            if actor.id == self.ego_vehicle.id:
                continue
            
            transform = actor.get_transform()
            velocity = actor.get_velocity()
            bbox = actor.bounding_box
            
            obj = {
                "id": actor.id,
                "position": np.array([
                    transform.location.x,
                    transform.location.y,
                    transform.location.z
                ]),
                "heading": np.radians(transform.rotation.yaw),
                "velocity": np.array([velocity.x, velocity.y, velocity.z]),
                "extent": np.array([bbox.extent.x, bbox.extent.y, bbox.extent.z]) * 2,  # Full dimensions
            }
            
            if "vehicle" in actor.type_id:
                gt["vehicles"].append(obj)
            elif "walker" in actor.type_id:
                gt["pedestrians"].append(obj)
        
        return gt


class SimulationRunner:
    """
    High-level simulation runner for closed-loop evaluation.
    """
    
    def __init__(
        self,
        client: CarlaClient,
        model: Any,
        controller: Any,
        max_steps: int = 1000,
        target_fps: float = 10.0,
    ):
        """
        Args:
            client: CARLA client
            model: Fusion2Drive model for inference
            controller: Controller for waypoint following
            max_steps: Maximum simulation steps
            target_fps: Target inference frequency
        """
        self.client = client
        self.model = model
        self.controller = controller
        self.max_steps = max_steps
        self.target_fps = target_fps
        
        self.metrics = {
            "collisions": 0,
            "distance_traveled": 0.0,
            "avg_speed": [],
            "waypoint_errors": [],
        }
    
    def run_episode(self) -> Dict[str, float]:
        """
        Run a single simulation episode.
        
        Returns:
            Dict with episode metrics
        """
        logger.info("Starting simulation episode...")
        
        last_position = None
        
        for step in range(self.max_steps):
            # Get sensor data
            sensor_data = self.client.get_sensor_data()
            
            if any(v is None for v in sensor_data.values()):
                logger.warning(f"Missing sensor data at step {step}")
                continue
            
            # Run model inference
            model_input = self._prepare_model_input(sensor_data)
            
            with torch.no_grad():
                outputs = self.model.inference(**model_input)
            
            waypoints = outputs["waypoints"][0].cpu().numpy()  # (T, 3)
            
            # Get ego state
            ego_state = self.client.get_ego_state()
            
            # Compute control
            throttle, steer, brake = self.controller.compute_control(
                current_position=ego_state["position"][:2],
                current_heading=ego_state["heading"],
                current_speed=ego_state["speed"],
                waypoints=waypoints[:, :2],
            )
            
            # Apply control
            self.client.apply_control(throttle=throttle, steer=steer, brake=brake)
            
            # Step simulation
            self.client.tick()
            
            # Update metrics
            if last_position is not None:
                distance = np.linalg.norm(ego_state["position"][:2] - last_position)
                self.metrics["distance_traveled"] += distance
            
            last_position = ego_state["position"][:2].copy()
            self.metrics["avg_speed"].append(ego_state["speed"])
            
            # Check for collisions
            # (simplified - would need collision sensor in full implementation)
        
        # Compute final metrics
        return {
            "distance_traveled": self.metrics["distance_traveled"],
            "avg_speed": np.mean(self.metrics["avg_speed"]) if self.metrics["avg_speed"] else 0.0,
            "num_steps": self.max_steps,
        }
    
    def _prepare_model_input(self, sensor_data: Dict) -> Dict:
        """Prepare sensor data for model inference."""
        import torch
        
        # Stack camera images
        camera_names = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]
        images = []
        
        for name in camera_names:
            img = sensor_data.get(f"camera_{name}")
            if img is not None:
                # Normalize and convert to tensor
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
                images.append(img)
        
        images = np.stack(images, axis=0)  # (N, 3, H, W)
        images = torch.from_numpy(images).unsqueeze(0)  # (1, N, 3, H, W)
        
        # Process LiDAR
        points = sensor_data.get("lidar")
        if points is not None:
            points = torch.from_numpy(points).unsqueeze(0)  # (1, P, 4)
        else:
            points = torch.zeros(1, 100000, 4)
        
        points_mask = torch.ones(1, points.shape[1], dtype=torch.bool)
        
        # Dummy calibration (would need proper intrinsics/extrinsics from CARLA)
        intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, 5, -1, -1)
        extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(1, 5, -1, -1)
        
        return {
            "images": images,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "points": points,
            "points_mask": points_mask,
        }


# Import torch for SimulationRunner
try:
    import torch
except ImportError:
    pass
