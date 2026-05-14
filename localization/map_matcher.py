"""
Map Matching Module

Matches LiDAR scans against known environment maps
for improved localization accuracy.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class EnvironmentMap:
    """
    Represents a known 2D environment map.
    
    Stores:
    - Occupancy grid
    - Wall positions
    - Landmark positions
    """
    
    def __init__(self, 
                 width_mm: float = 3000.0,
                 height_mm: float = 2000.0,
                 grid_cell_mm: float = 50.0):
        """
        Initialize environment map.
        
        Args:
            width_mm: Map width in millimeters
            height_mm: Map height in millimeters
            grid_cell_mm: Size of occupancy grid cells
        """
        self.width_mm = width_mm
        self.height_mm = height_mm
        self.grid_cell_mm = grid_cell_mm
        
        # Occupancy grid: 0=free, 1=occupied
        self.grid_cols = int(width_mm / grid_cell_mm)
        self.grid_rows = int(height_mm / grid_cell_mm)
        self.occupancy_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.uint8)
        
        # Known features
        self.walls: List[Dict] = []
        self.landmarks: Dict[str, Tuple[float, float]] = {}
        
        logger.info(f"EnvironmentMap created: {width_mm}×{height_mm}mm, "
                   f"grid {self.grid_cols}×{self.grid_rows} @ {grid_cell_mm}mm/cell")
    
    def add_wall(self, x1_mm: float, y1_mm: float, x2_mm: float, y2_mm: float):
        """
        Add wall segment to map.
        
        Args:
            x1_mm, y1_mm: Start point in mm
            x2_mm, y2_mm: End point in mm
        """
        wall = {
            "x1": x1_mm,
            "y1": y1_mm,
            "x2": x2_mm,
            "y2": y2_mm,
            "length_mm": np.sqrt((x2_mm-x1_mm)**2 + (y2_mm-y1_mm)**2)
        }
        self.walls.append(wall)
        
        # Mark occupancy grid
        self._mark_line_occupied(x1_mm, y1_mm, x2_mm, y2_mm)
    
    def add_landmark(self, name: str, x_mm: float, y_mm: float):
        """
        Add known landmark to map.
        
        Args:
            name: Landmark identifier
            x_mm: X position in mm
            y_mm: Y position in mm
        """
        self.landmarks[name] = (x_mm, y_mm)
    
    def add_obstacles_from_points(self, points: np.ndarray, threshold_count: int = 3):
        """
        Add obstacles detected from point cloud.
        
        Args:
            points: Point cloud array (N, 2)
            threshold_count: Minimum points to create obstacle
        """
        if len(points) < threshold_count:
            return
        
        # Mark occupied cells
        for x, y in points:
            col = int(x / self.grid_cell_mm)
            row = int(y / self.grid_cell_mm)
            if 0 <= col < self.grid_cols and 0 <= row < self.grid_rows:
                self.occupancy_grid[row, col] = 1
    
    def _mark_line_occupied(self, x1: float, y1: float, x2: float, y2: float):
        """Mark cells along a line as occupied (Bresenham)."""
        # Convert to grid coordinates
        c1 = int(x1 / self.grid_cell_mm)
        r1 = int(y1 / self.grid_cell_mm)
        c2 = int(x2 / self.grid_cell_mm)
        r2 = int(y2 / self.grid_cell_mm)
        
        # Bresenham line algorithm
        dx = abs(c2 - c1)
        dy = abs(r2 - r1)
        sx = 1 if c1 < c2 else -1
        sy = 1 if r1 < r2 else -1
        
        if dx > dy:
            err = dx / 2
            r = r1
            for c in range(c1, c2 + sx, sx):
                if 0 <= c < self.grid_cols and 0 <= r < self.grid_rows:
                    self.occupancy_grid[r, c] = 1
                err -= dy
                if err < 0:
                    r += sy
                    err += dx
        else:
            err = dy / 2
            c = c1
            for r in range(r1, r2 + sy, sy):
                if 0 <= c < self.grid_cols and 0 <= r < self.grid_rows:
                    self.occupancy_grid[r, c] = 1
                err -= dx
                if err < 0:
                    c += sx
                    err += dy
    
    def get_occupancy_at(self, x_mm: float, y_mm: float) -> int:
        """Get occupancy value at position (0=free, 1=occupied)."""
        col = int(x_mm / self.grid_cell_mm)
        row = int(y_mm / self.grid_cell_mm)
        
        if 0 <= col < self.grid_cols and 0 <= row < self.grid_rows:
            return int(self.occupancy_grid[row, col])
        return 0  # Outside map = free
    
    def to_dict(self) -> Dict:
        """Serialize map to dictionary."""
        return {
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "grid_cell_mm": self.grid_cell_mm,
            "walls": self.walls,
            "landmarks": self.landmarks,
            "grid": self.occupancy_grid.tolist()
        }
    
    @staticmethod
    def from_dict(data: Dict) -> EnvironmentMap:
        """Deserialize map from dictionary."""
        env_map = EnvironmentMap(
            data["width_mm"],
            data["height_mm"],
            data["grid_cell_mm"]
        )
        env_map.walls = data.get("walls", [])
        env_map.landmarks = data.get("landmarks", {})
        env_map.occupancy_grid = np.array(data.get("grid", []), dtype=np.uint8)
        return env_map


class MapMatcher:
    """
    Matches LiDAR scans to known environment maps.
    
    Uses occupancy grid similarity to estimate pose corrections.
    """
    
    def __init__(self, environment_map: EnvironmentMap):
        """
        Initialize map matcher.
        
        Args:
            environment_map: Known environment map
        """
        self.map = environment_map
        self.last_match_result = None
        logger.info("MapMatcher initialized")
    
    def match_scan(self,
                   scan_points: np.ndarray,
                   robot_pose: Tuple[float, float, float],
                   search_radius_mm: float = 200.0,
                   angle_step_deg: float = 5.0) -> Dict:
        """
        Match LiDAR scan to known map.
        
        Uses a grid search to find best matching position and orientation.
        
        Args:
            scan_points: Point cloud from LiDAR (N, 2) in robot frame
            robot_pose: Current pose estimate (x, y, theta)
            search_radius_mm: Search radius around current pose
            angle_step_deg: Angle step for rotation search
        
        Returns:
            Dictionary with match results:
            - best_match: Best matching pose
            - confidence: Match confidence (0-1)
            - scores: Score at each search position
        """
        if len(scan_points) == 0:
            return {
                "best_match": robot_pose,
                "confidence": 0.0,
                "reason": "empty_scan"
            }
        
        robot_x, robot_y, robot_theta = robot_pose
        best_score = -np.inf
        best_pose = robot_pose
        
        # Grid search over position
        x_offsets = np.arange(-search_radius_mm, search_radius_mm + 1, 50)
        y_offsets = np.arange(-search_radius_mm, search_radius_mm + 1, 50)
        angle_offsets = np.arange(-30, 31, angle_step_deg)
        
        scores = []
        
        for dx in x_offsets:
            for dy in y_offsets:
                for dtheta in angle_offsets:
                    # Transform scan to world frame
                    theta_rad = np.radians(robot_theta + dtheta)
                    cos_t = np.cos(theta_rad)
                    sin_t = np.sin(theta_rad)
                    
                    # Rotate and translate scan points
                    rotated = np.zeros_like(scan_points)
                    rotated[:, 0] = scan_points[:, 0] * cos_t - scan_points[:, 1] * sin_t
                    rotated[:, 1] = scan_points[:, 0] * sin_t + scan_points[:, 1] * cos_t
                    
                    world_points = rotated + np.array([[robot_x + dx, robot_y + dy]])
                    
                    # Score: how many scan points match map occupancy
                    score = self._score_match(world_points)
                    scores.append(score)
                    
                    if score > best_score:
                        best_score = score
                        best_pose = (robot_x + dx, robot_y + dy, robot_theta + dtheta)
        
        # Normalize confidence
        scores = np.array(scores)
        if len(scores) > 0:
            confidence = (best_score - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-6)
        else:
            confidence = 0.0
        
        self.last_match_result = {
            "best_match": best_pose,
            "confidence": float(confidence),
            "score": float(best_score)
        }
        
        return self.last_match_result
    
    def _score_match(self, world_points: np.ndarray) -> float:
        """
        Score how well scan points match map occupancy.
        
        Returns:
            Score (higher = better match)
        """
        if len(world_points) == 0:
            return 0.0
        
        score = 0.0
        for x, y in world_points:
            occupancy = self.map.get_occupancy_at(x, y)
            # Reward: points in free space get +1, in occupied get -1
            score += (1.0 if occupancy == 0 else -0.5)
        
        return score / len(world_points)
    
    def find_landmarks_in_scan(self,
                               scan_points: np.ndarray,
                               robot_pose: Tuple[float, float, float],
                               detection_radius_mm: float = 500.0) -> Dict[str, Dict]:
        """
        Find known landmarks in current scan.
        
        Args:
            scan_points: Point cloud from LiDAR
            robot_pose: Current robot pose
            detection_radius_mm: Search radius for landmarks
        
        Returns:
            Dictionary mapping landmark names to detection info
        """
        robot_x, robot_y, robot_theta = robot_pose
        detected = {}
        
        for landmark_name, (land_x, land_y) in self.map.landmarks.items():
            # Is landmark visible from robot?
            dist_to_landmark = np.sqrt((land_x - robot_x)**2 + (land_y - robot_y)**2)
            
            if dist_to_landmark < detection_radius_mm:
                # Check if landmark is in scan points
                distances = np.linalg.norm(scan_points - np.array([land_x - robot_x, land_y - robot_y]), axis=1)
                
                if np.min(distances) < 100:  # Close to a point in scan
                    detected[landmark_name] = {
                        "distance_mm": float(dist_to_landmark),
                        "closest_point_distance": float(np.min(distances)),
                        "scan_point_index": int(np.argmin(distances))
                    }
        
        return detected
