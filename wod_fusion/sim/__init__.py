"""
CARLA simulation module for closed-loop evaluation.
"""

from wod_fusion.sim.carla_client import CarlaClient, CarlaConfig
from wod_fusion.sim.controllers import PurePursuitController, PIDController, MPCController
from wod_fusion.sim.sensor_interface import SensorInterface, SensorConfig

__all__ = [
    "CarlaClient",
    "CarlaConfig",
    "PurePursuitController",
    "PIDController",
    "MPCController",
    "SensorInterface",
    "SensorConfig",
]
