"""
Pose Estimation Module

Estimates robot position and orientation using:
- Odometry data
- LiDAR scan matching
- IMU orientation (if available)
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Pose:
    """Robot pose in 2D space."""
    x_mm: float = 0.0          # X position in millimeters
    y_mm: float = 0.0          # Y position in millimeters
    theta_deg: float = 0.0     # Rotation in degrees (0-360)
    timestamp: float = 0.0     # Unix timestamp
    confidence: float = 1.0    # Confidence (0-1)
    source: str = "init"       # Source: "init", "odom", "lidar", "fused"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class PoseEstimator:
    """
    Estimates and tracks robot pose using sensor fusion.
    
    Combines odometry with LiDAR scan matching for robust positioning.
    """
    
    def __init__(self, 
                 init_x_mm: float = 0.0,
                 init_y_mm: float = 0.0,
                 init_theta_deg: float = 0.0):
        """
        Initialize pose estimator.
        
        Args:
            init_x_mm: Initial X position in mm
            init_y_mm: Initial Y position in mm
            init_theta_deg: Initial orientation in degrees
        """
        now = datetime.now().timestamp()
        self.pose = Pose(
            x_mm=init_x_mm,
            y_mm=init_y_mm,
            theta_deg=init_theta_deg,
            timestamp=now,
            source="init"
        )
        
        # Odometry tracking
        self.last_odom_pose = None
        self.odom_drift_x = 0.0
        self.odom_drift_y = 0.0
        self.odom_drift_theta = 0.0
        
        # LiDAR scan matching
        self.last_scan_points = None
        self.scan_match_confidence = 0.0
        
        logger.info(f"PoseEstimator initialized at ({init_x_mm}, {init_y_mm}, {init_theta_deg}°)")
    
    def update_from_odom(self, 
                         x_mm: float, 
                         y_mm: float, 
                         theta_deg: float) -> Pose:
        """
        Update pose from odometry data.
        
        Args:
            x_mm: Position X from odometry
            y_mm: Position Y from odometry
            theta_deg: Orientation from odometry
        
        Returns:
            Updated pose
        """
        now = datetime.now().timestamp()
        
        # Track odometry drift for later correction
        if self.last_odom_pose is not None:
            dt = now - self.last_odom_pose['timestamp']
            dx = x_mm - self.last_odom_pose['x']
            dy = y_mm - self.last_odom_pose['y']
            dtheta = theta_deg - self.last_odom_pose['theta']
            
            # Normalize angle difference
            while dtheta > 180:
                dtheta -= 360
            while dtheta < -180:
                dtheta += 360
            
            # Update odom drift
            self.odom_drift_x += abs(dx)
            self.odom_drift_y += abs(dy)
            self.odom_drift_theta += abs(dtheta)
        
        self.last_odom_pose = {
            'x': x_mm,
            'y': y_mm,
            'theta': theta_deg,
            'timestamp': now
        }
        
        # Simple: use odometry directly (can be refined with scan matching)
        self.pose = Pose(
            x_mm=x_mm,
            y_mm=y_mm,
            theta_deg=self._normalize_angle(theta_deg),
            timestamp=now,
            confidence=max(0.5, 1.0 - min(0.5, self.odom_drift_x / 1000.0)),  # Confidence decreases with drift
            source="odom"
        )
        
        return self.pose
    
    def update_from_scan_match(self, 
                               delta_x_mm: float, 
                               delta_y_mm: float, 
                               delta_theta_deg: float,
                               confidence: float = 0.8) -> Pose:
        """
        Correct pose using LiDAR scan matching results.
        
        Args:
            delta_x_mm: Correction to X position
            delta_y_mm: Correction to Y position
            delta_theta_deg: Correction to orientation
            confidence: Confidence of scan matching (0-1)
        
        Returns:
            Corrected pose
        """
        now = datetime.now().timestamp()
        
        # Apply correction with confidence weighting
        self.pose.x_mm += delta_x_mm * confidence
        self.pose.y_mm += delta_y_mm * confidence
        self.pose.theta_deg = self._normalize_angle(
            self.pose.theta_deg + delta_theta_deg * confidence
        )
        self.pose.timestamp = now
        self.pose.confidence = min(0.99, self.pose.confidence * 0.8 + confidence * 0.2)
        self.pose.source = "fused"
        
        self.scan_match_confidence = confidence
        
        return self.pose
    
    def reset_drift(self):
        """Reset odometry drift counters after localization reset."""
        self.odom_drift_x = 0.0
        self.odom_drift_y = 0.0
        self.odom_drift_theta = 0.0
        logger.info("Odometry drift reset")
    
    def set_absolute_position(self, x_mm: float, y_mm: float, theta_deg: float):
        """
        Set absolute position (e.g., from external localization).
        
        Args:
            x_mm: Absolute X position in mm
            y_mm: Absolute Y position in mm
            theta_deg: Absolute orientation in degrees
        """
        self.pose = Pose(
            x_mm=x_mm,
            y_mm=y_mm,
            theta_deg=self._normalize_angle(theta_deg),
            timestamp=datetime.now().timestamp(),
            confidence=0.95,
            source="absolute"
        )
        self.reset_drift()
        logger.info(f"Absolute position set: ({x_mm}, {y_mm}, {theta_deg}°)")
    
    def get_pose(self) -> Pose:
        """Get current pose estimate."""
        return self.pose
    
    def get_pose_dict(self) -> Dict:
        """Get current pose as dictionary."""
        return self.pose.to_dict()
    
    def get_position_mm(self) -> Tuple[float, float]:
        """Get (x, y) position in millimeters."""
        return (self.pose.x_mm, self.pose.y_mm)
    
    def get_orientation_deg(self) -> float:
        """Get orientation in degrees (0-360)."""
        return self.pose.theta_deg
    
    def get_distance_to_point(self, x_mm: float, y_mm: float) -> float:
        """
        Calculate distance to a point.
        
        Args:
            x_mm: Target X in mm
            y_mm: Target Y in mm
        
        Returns:
            Distance in millimeters
        """
        dx = x_mm - self.pose.x_mm
        dy = y_mm - self.pose.y_mm
        return np.sqrt(dx**2 + dy**2)
    
    def get_bearing_to_point(self, x_mm: float, y_mm: float) -> float:
        """
        Calculate bearing (angle) to a point.
        
        Args:
            x_mm: Target X in mm
            y_mm: Target Y in mm
        
        Returns:
            Bearing in degrees (relative to robot orientation)
        """
        dx = x_mm - self.pose.x_mm
        dy = y_mm - self.pose.y_mm
        target_angle = np.degrees(np.arctan2(dy, dx))
        bearing = target_angle - self.pose.theta_deg
        
        # Normalize to -180 to 180
        while bearing > 180:
            bearing -= 360
        while bearing < -180:
            bearing += 360
        
        return bearing
    
    def _normalize_angle(self, angle_deg: float) -> float:
        """Normalize angle to 0-360 range."""
        angle = angle_deg % 360
        if angle < 0:
            angle += 360
        return angle
    
    def get_drift_estimate(self) -> Dict:
        """Get estimated odometry drift."""
        return {
            "drift_x_mm": self.odom_drift_x,
            "drift_y_mm": self.odom_drift_y,
            "drift_theta_deg": self.odom_drift_theta,
            "total_drift_mm": np.sqrt(self.odom_drift_x**2 + self.odom_drift_y**2),
            "scan_match_confidence": self.scan_match_confidence
        }
