"""
Vehicle controllers for waypoint following.

Implements Pure Pursuit, PID, and MPC controllers.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class BaseController(ABC):
    """Abstract base class for vehicle controllers."""
    
    @abstractmethod
    def compute_control(
        self,
        current_position: np.ndarray,
        current_heading: float,
        current_speed: float,
        waypoints: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Compute vehicle control commands.
        
        Args:
            current_position: (2,) current x, y position
            current_heading: Current heading in radians
            current_speed: Current speed in m/s
            waypoints: (T, 2) future waypoints in world frame
            
        Returns:
            throttle, steer, brake (all in [0, 1] or [-1, 1] for steer)
        """
        pass
    
    def reset(self):
        """Reset controller state."""
        pass


@dataclass
class PurePursuitConfig:
    """Configuration for Pure Pursuit controller."""
    
    lookahead_distance: float = 5.0  # meters
    min_lookahead: float = 2.0
    max_lookahead: float = 15.0
    lookahead_gain: float = 0.3  # Scales with speed
    
    wheelbase: float = 2.9  # meters (typical sedan)
    max_steer: float = 0.5  # radians
    
    target_speed: float = 10.0  # m/s (36 km/h)
    min_speed: float = 1.0
    max_speed: float = 20.0
    
    # PID gains for speed control
    kp_speed: float = 0.5
    ki_speed: float = 0.1
    kd_speed: float = 0.05


class PurePursuitController(BaseController):
    """
    Pure Pursuit controller for path following.
    
    Computes steering angle to track a lookahead point on the path.
    Uses PID for longitudinal (speed) control.
    """
    
    def __init__(self, config: Optional[PurePursuitConfig] = None):
        self.config = config or PurePursuitConfig()
        
        # Speed PID state
        self.speed_integral = 0.0
        self.prev_speed_error = 0.0
    
    def compute_control(
        self,
        current_position: np.ndarray,
        current_heading: float,
        current_speed: float,
        waypoints: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Compute control using Pure Pursuit algorithm."""
        
        if len(waypoints) == 0:
            return 0.0, 0.0, 1.0  # Emergency stop
        
        # Compute adaptive lookahead distance
        lookahead = self.config.min_lookahead + self.config.lookahead_gain * current_speed
        lookahead = np.clip(lookahead, self.config.min_lookahead, self.config.max_lookahead)
        
        # Find lookahead point on path
        lookahead_point = self._find_lookahead_point(
            current_position, waypoints, lookahead
        )
        
        # Compute steering using Pure Pursuit geometry
        steer = self._compute_steering(
            current_position, current_heading, lookahead_point, lookahead
        )
        
        # Compute throttle/brake using PID speed control
        # Reduce target speed for sharp turns
        curvature = abs(steer) / self.config.wheelbase
        speed_factor = 1.0 / (1.0 + 2.0 * curvature)  # Slow down for curves
        target_speed = self.config.target_speed * speed_factor
        target_speed = max(target_speed, self.config.min_speed)
        
        throttle, brake = self._compute_longitudinal(current_speed, target_speed)
        
        return throttle, steer, brake
    
    def _find_lookahead_point(
        self,
        current_position: np.ndarray,
        waypoints: np.ndarray,
        lookahead: float,
    ) -> np.ndarray:
        """Find point on path at lookahead distance."""
        
        # Compute distances to all waypoints
        distances = np.linalg.norm(waypoints - current_position, axis=1)
        
        # Find first waypoint beyond lookahead
        beyond_mask = distances >= lookahead
        
        if np.any(beyond_mask):
            # Interpolate to exact lookahead distance
            idx = np.argmax(beyond_mask)
            if idx == 0:
                return waypoints[0]
            
            # Linear interpolation
            d1 = distances[idx - 1]
            d2 = distances[idx]
            t = (lookahead - d1) / (d2 - d1 + 1e-6)
            t = np.clip(t, 0.0, 1.0)
            
            return waypoints[idx - 1] + t * (waypoints[idx] - waypoints[idx - 1])
        else:
            # All waypoints are closer, use the farthest one
            return waypoints[-1]
    
    def _compute_steering(
        self,
        current_position: np.ndarray,
        current_heading: float,
        lookahead_point: np.ndarray,
        lookahead_distance: float,
    ) -> float:
        """Compute steering angle using Pure Pursuit geometry."""
        
        # Vector from current position to lookahead point
        dx = lookahead_point[0] - current_position[0]
        dy = lookahead_point[1] - current_position[1]
        
        # Transform to vehicle frame
        cos_h = np.cos(current_heading)
        sin_h = np.sin(current_heading)
        
        # Lateral offset in vehicle frame
        lateral_error = -dx * sin_h + dy * cos_h
        
        # Pure Pursuit steering law
        # curvature = 2 * lateral_error / L^2
        # steer = atan(curvature * wheelbase)
        actual_lookahead = np.sqrt(dx**2 + dy**2)
        curvature = 2 * lateral_error / (actual_lookahead**2 + 1e-6)
        steer = np.arctan(curvature * self.config.wheelbase)
        
        # Clip to max steering angle
        steer = np.clip(steer, -self.config.max_steer, self.config.max_steer)
        
        # Normalize to [-1, 1] for control interface
        return steer / self.config.max_steer
    
    def _compute_longitudinal(
        self,
        current_speed: float,
        target_speed: float,
    ) -> Tuple[float, float]:
        """Compute throttle and brake using PID control."""
        
        speed_error = target_speed - current_speed
        
        # PID control
        self.speed_integral += speed_error
        self.speed_integral = np.clip(self.speed_integral, -10.0, 10.0)
        
        speed_derivative = speed_error - self.prev_speed_error
        self.prev_speed_error = speed_error
        
        control = (
            self.config.kp_speed * speed_error +
            self.config.ki_speed * self.speed_integral +
            self.config.kd_speed * speed_derivative
        )
        
        if control >= 0:
            throttle = np.clip(control, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(-control, 0.0, 1.0)
        
        return throttle, brake
    
    def reset(self):
        """Reset controller state."""
        self.speed_integral = 0.0
        self.prev_speed_error = 0.0


@dataclass
class PIDConfig:
    """Configuration for PID controller."""
    
    # Lateral PID gains
    kp_lateral: float = 0.5
    ki_lateral: float = 0.01
    kd_lateral: float = 0.1
    
    # Heading PID gains
    kp_heading: float = 1.0
    ki_heading: float = 0.01
    kd_heading: float = 0.2
    
    # Speed PID gains
    kp_speed: float = 0.5
    ki_speed: float = 0.1
    kd_speed: float = 0.05
    
    wheelbase: float = 2.9
    max_steer: float = 0.5
    target_speed: float = 10.0


class PIDController(BaseController):
    """
    PID-based controller with separate lateral and heading control.
    
    More aggressive tracking than Pure Pursuit, suitable for
    precise waypoint following.
    """
    
    def __init__(self, config: Optional[PIDConfig] = None):
        self.config = config or PIDConfig()
        
        # PID state
        self.lateral_integral = 0.0
        self.prev_lateral_error = 0.0
        
        self.heading_integral = 0.0
        self.prev_heading_error = 0.0
        
        self.speed_integral = 0.0
        self.prev_speed_error = 0.0
    
    def compute_control(
        self,
        current_position: np.ndarray,
        current_heading: float,
        current_speed: float,
        waypoints: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Compute control using cascaded PID."""
        
        if len(waypoints) < 2:
            return 0.0, 0.0, 1.0
        
        # Get target point (first waypoint)
        target = waypoints[0]
        
        # Compute path heading from first two waypoints
        path_heading = np.arctan2(
            waypoints[1][1] - waypoints[0][1],
            waypoints[1][0] - waypoints[0][0]
        )
        
        # Lateral error (cross-track error)
        dx = target[0] - current_position[0]
        dy = target[1] - current_position[1]
        
        cos_h = np.cos(current_heading)
        sin_h = np.sin(current_heading)
        lateral_error = -dx * sin_h + dy * cos_h
        
        # Heading error
        heading_error = self._normalize_angle(path_heading - current_heading)
        
        # Lateral PID
        self.lateral_integral += lateral_error
        self.lateral_integral = np.clip(self.lateral_integral, -5.0, 5.0)
        lateral_derivative = lateral_error - self.prev_lateral_error
        self.prev_lateral_error = lateral_error
        
        lateral_control = (
            self.config.kp_lateral * lateral_error +
            self.config.ki_lateral * self.lateral_integral +
            self.config.kd_lateral * lateral_derivative
        )
        
        # Heading PID
        self.heading_integral += heading_error
        self.heading_integral = np.clip(self.heading_integral, -2.0, 2.0)
        heading_derivative = heading_error - self.prev_heading_error
        self.prev_heading_error = heading_error
        
        heading_control = (
            self.config.kp_heading * heading_error +
            self.config.ki_heading * self.heading_integral +
            self.config.kd_heading * heading_derivative
        )
        
        # Combine lateral and heading control
        steer = lateral_control + heading_control
        steer = np.clip(steer / self.config.max_steer, -1.0, 1.0)
        
        # Speed control
        speed_error = self.config.target_speed - current_speed
        
        self.speed_integral += speed_error
        self.speed_integral = np.clip(self.speed_integral, -10.0, 10.0)
        speed_derivative = speed_error - self.prev_speed_error
        self.prev_speed_error = speed_error
        
        speed_control = (
            self.config.kp_speed * speed_error +
            self.config.ki_speed * self.speed_integral +
            self.config.kd_speed * speed_derivative
        )
        
        if speed_control >= 0:
            throttle = np.clip(speed_control, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(-speed_control, 0.0, 1.0)
        
        return throttle, steer, brake
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def reset(self):
        """Reset controller state."""
        self.lateral_integral = 0.0
        self.prev_lateral_error = 0.0
        self.heading_integral = 0.0
        self.prev_heading_error = 0.0
        self.speed_integral = 0.0
        self.prev_speed_error = 0.0


@dataclass
class MPCConfig:
    """Configuration for MPC controller."""
    
    horizon: int = 10
    dt: float = 0.1  # Time step
    
    wheelbase: float = 2.9
    max_steer: float = 0.5
    max_steer_rate: float = 0.3  # rad/s
    max_accel: float = 3.0  # m/s^2
    max_decel: float = 5.0  # m/s^2
    
    target_speed: float = 10.0
    
    # Cost weights
    w_position: float = 1.0
    w_heading: float = 0.5
    w_speed: float = 0.3
    w_steer: float = 0.1
    w_steer_rate: float = 0.5
    w_accel: float = 0.1


class MPCController(BaseController):
    """
    Model Predictive Control for trajectory tracking.
    
    Uses a simplified kinematic bicycle model and quadratic
    optimization for smooth, optimal control.
    
    Note: This is a simplified MPC. For real-time performance,
    consider using CVXPY or CasADi with IPOPT.
    """
    
    def __init__(self, config: Optional[MPCConfig] = None):
        self.config = config or MPCConfig()
        
        self.prev_steer = 0.0
        self.prev_accel = 0.0
    
    def compute_control(
        self,
        current_position: np.ndarray,
        current_heading: float,
        current_speed: float,
        waypoints: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Compute control using MPC."""
        
        if len(waypoints) < self.config.horizon:
            # Pad waypoints if not enough
            last_wp = waypoints[-1] if len(waypoints) > 0 else current_position
            padding = np.tile(last_wp, (self.config.horizon - len(waypoints), 1))
            waypoints = np.vstack([waypoints, padding]) if len(waypoints) > 0 else padding
        
        # Simplified MPC: Use iterative LQR approach
        # For production, use proper QP solver
        
        best_steer = 0.0
        best_cost = float("inf")
        
        # Grid search over steering angles
        steer_candidates = np.linspace(-1.0, 1.0, 21)
        
        for steer in steer_candidates:
            steer_rad = steer * self.config.max_steer
            
            # Simulate trajectory
            traj = self._simulate_trajectory(
                current_position, current_heading, current_speed, steer_rad
            )
            
            # Compute cost
            cost = self._compute_trajectory_cost(traj, waypoints, steer_rad)
            
            if cost < best_cost:
                best_cost = cost
                best_steer = steer
        
        # Compute throttle/brake based on predicted speed
        if current_speed < self.config.target_speed:
            throttle = np.clip(
                (self.config.target_speed - current_speed) / self.config.max_accel,
                0.0, 1.0
            )
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(
                (current_speed - self.config.target_speed) / self.config.max_decel,
                0.0, 1.0
            )
        
        self.prev_steer = best_steer
        
        return throttle, best_steer, brake
    
    def _simulate_trajectory(
        self,
        start_position: np.ndarray,
        start_heading: float,
        start_speed: float,
        steer: float,
    ) -> np.ndarray:
        """Simulate trajectory with constant steering."""
        
        trajectory = np.zeros((self.config.horizon, 3))  # x, y, heading
        
        x, y, heading = start_position[0], start_position[1], start_heading
        speed = start_speed
        
        for t in range(self.config.horizon):
            # Bicycle model kinematics
            x += speed * np.cos(heading) * self.config.dt
            y += speed * np.sin(heading) * self.config.dt
            heading += (speed / self.config.wheelbase) * np.tan(steer) * self.config.dt
            
            trajectory[t] = [x, y, heading]
        
        return trajectory
    
    def _compute_trajectory_cost(
        self,
        trajectory: np.ndarray,
        waypoints: np.ndarray,
        steer: float,
    ) -> float:
        """Compute trajectory cost."""
        
        # Position error
        pos_error = np.sum((trajectory[:, :2] - waypoints[:self.config.horizon])**2)
        
        # Steering cost
        steer_cost = steer**2
        
        # Steering rate cost
        steer_rate = steer - self.prev_steer * self.config.max_steer
        steer_rate_cost = steer_rate**2
        
        total_cost = (
            self.config.w_position * pos_error +
            self.config.w_steer * steer_cost +
            self.config.w_steer_rate * steer_rate_cost
        )
        
        return total_cost
    
    def reset(self):
        """Reset controller state."""
        self.prev_steer = 0.0
        self.prev_accel = 0.0


def create_controller(controller_type: str, **kwargs) -> BaseController:
    """
    Factory function to create controllers.
    
    Args:
        controller_type: One of "pure_pursuit", "pid", "mpc"
        **kwargs: Config parameters
        
    Returns:
        Controller instance
    """
    controllers = {
        "pure_pursuit": (PurePursuitController, PurePursuitConfig),
        "pid": (PIDController, PIDConfig),
        "mpc": (MPCController, MPCConfig),
    }
    
    if controller_type not in controllers:
        raise ValueError(f"Unknown controller type: {controller_type}")
    
    controller_cls, config_cls = controllers[controller_type]
    config = config_cls(**kwargs)
    
    return controller_cls(config)
