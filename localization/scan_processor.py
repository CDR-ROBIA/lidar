"""
Scan Processing Module

Processes raw LiDAR scan data and converts it to point clouds
in the robot's coordinate system.
"""

from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ScanProcessor:
    """
    Processes raw LiDAR scan data from RPLidar format.
    
    Converts polar coordinates (angle, distance) to Cartesian (x, y)
    in the robot's local coordinate system.
    """
    
    def __init__(self, 
                 max_range_mm: float = 6000.0,
                 min_range_mm: float = 50.0,
                 quality_threshold: int = 5):
        """
        Initialize scan processor.
        
        Args:
            max_range_mm: Maximum valid range in millimeters
            min_range_mm: Minimum valid range in millimeters
            quality_threshold: Minimum quality value (0-255) to accept point
        """
        self.max_range_mm = max_range_mm
        self.min_range_mm = min_range_mm
        self.quality_threshold = quality_threshold
    
    def process_scan(self, scan: List[Tuple[int, float, float]]) -> np.ndarray:
        """
        Convert raw scan to point cloud.
        
        Args:
            scan: List of (quality, angle_deg, distance_mm) tuples
        
        Returns:
            numpy array of shape (N, 2) with (x, y) points in mm
        """
        if not scan:
            return np.empty((0, 2), dtype=np.float32)
        
        points = []
        
        for quality, angle_deg, distance_mm in scan:
            # Filter by quality
            if quality < self.quality_threshold:
                continue
            
            # Filter by range
            if distance_mm < self.min_range_mm or distance_mm > self.max_range_mm:
                continue
            
            # Convert polar to Cartesian
            angle_rad = np.radians(angle_deg)
            x = distance_mm * np.cos(angle_rad)
            y = distance_mm * np.sin(angle_rad)
            
            points.append([x, y])
        
        if not points:
            return np.empty((0, 2), dtype=np.float32)
        
        return np.array(points, dtype=np.float32)
    
    def filter_noise(self, points: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Filter outliers from point cloud using statistical method.
        
        Args:
            points: Point cloud array of shape (N, 2)
            window_size: Size of neighborhood for filtering
        
        Returns:
            Filtered point cloud
        """
        if len(points) < window_size:
            return points
        
        # Simple distance-based filtering: remove points far from neighbors
        filtered = []
        for i, point in enumerate(points):
            # Calculate distances to all other points
            distances = np.linalg.norm(points - point, axis=1)
            
            # Keep point if it has neighbors within threshold
            nearby = np.sum(distances < 100.0)  # 100mm neighborhood
            if nearby >= 2:  # At least 2 neighbors including self
                filtered.append(point)
        
        return np.array(filtered, dtype=np.float32) if filtered else np.empty((0, 2), dtype=np.float32)
    
    def detect_walls(self, points: np.ndarray, angle_tolerance_deg: float = 5.0) -> List[Dict]:
        """
        Detect wall/obstacle segments from scan points.
        
        Args:
            points: Point cloud array
            angle_tolerance_deg: Tolerance for wall angle detection
        
        Returns:
            List of detected wall segments with positions and angles
        """
        if len(points) < 3:
            return []
        
        walls = []
        
        # Group points by proximity to detect walls
        processed = set()
        
        for i, point in enumerate(points):
            if i in processed:
                continue
            
            # Find connected points (within ~50mm)
            cluster = [point]
            for j in range(i + 1, len(points)):
                if j not in processed:
                    dist = np.linalg.norm(points[j] - point)
                    if dist < 100.0:  # 100mm cluster threshold
                        cluster.append(points[j])
                        processed.add(j)
            
            if len(cluster) >= 2:
                cluster_arr = np.array(cluster)
                
                # Fit line to cluster (simple wall detection)
                # Use PCA to find dominant direction
                mean = cluster_arr.mean(axis=0)
                cov = np.cov(cluster_arr.T)
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                dominant_dir = eigenvectors[:, 0]
                
                # Wall angle in degrees
                wall_angle = np.degrees(np.arctan2(dominant_dir[1], dominant_dir[0]))
                
                wall = {
                    "center_x": float(mean[0]),
                    "center_y": float(mean[1]),
                    "angle_deg": float(wall_angle),
                    "length_mm": float(np.max(np.linalg.norm(cluster_arr - mean, axis=1)) * 2),
                    "points_count": len(cluster)
                }
                walls.append(wall)
        
        return walls
    
    def get_statistics(self, scan: List[Tuple[int, float, float]]) -> Dict:
        """
        Calculate statistics about scan quality and coverage.
        
        Args:
            scan: Raw scan data
        
        Returns:
            Dictionary with statistics
        """
        if not scan:
            return {
                "total_points": 0,
                "valid_points": 0,
                "avg_distance_mm": 0.0,
                "avg_quality": 0,
                "angle_coverage": 0.0
            }
        
        qualities = []
        distances = []
        angles = []
        
        for quality, angle_deg, distance_mm in scan:
            if self.min_range_mm <= distance_mm <= self.max_range_mm:
                qualities.append(quality)
                distances.append(distance_mm)
                angles.append(angle_deg)
        
        return {
            "total_points": len(scan),
            "valid_points": len(distances),
            "avg_distance_mm": float(np.mean(distances)) if distances else 0.0,
            "min_distance_mm": float(np.min(distances)) if distances else 0.0,
            "max_distance_mm": float(np.max(distances)) if distances else 0.0,
            "avg_quality": float(np.mean(qualities)) if qualities else 0,
            "angle_coverage": float(np.ptp(angles)) if angles else 0.0  # Peak-to-peak
        }
