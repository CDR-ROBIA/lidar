"""
Integration Example

Shows how to integrate the localization system with the existing
LiDAR acquisition system from lidar-main.
"""

from __future__ import annotations
import logging
import time
from typing import Optional

# LiDAR system
from lidar.lidar import Lidar, find_lidar_port

# Localization system
from localization.pose_estimator import PoseEstimator
from localization.scan_processor import ScanProcessor
from localization.map_matcher import EnvironmentMap, MapMatcher

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class LocalizedRobot:
    """
    Complete robot system with localization.
    
    Integrates:
    - LiDAR hardware acquisition
    - Pose estimation
    - Scan matching
    - Position tracking
    """
    
    def __init__(self, 
                 lidar_port: Optional[str] = None,
                 init_x_mm: float = 1500.0,
                 init_y_mm: float = 1000.0,
                 init_theta_deg: float = 0.0):
        """
        Initialize localized robot.
        
        Args:
            lidar_port: Serial port for LiDAR (auto-detect if None)
            init_x_mm: Starting X position in mm
            init_y_mm: Starting Y position in mm
            init_theta_deg: Starting orientation in degrees
        """
        # Find and initialize LiDAR
        if lidar_port is None:
            lidar_port = find_lidar_port()
        
        logger.info(f"LiDAR port: {lidar_port}")
        self.lidar = Lidar(port=lidar_port)
        
        # Initialize localization components
        self.pose_estimator = PoseEstimator(init_x_mm, init_y_mm, init_theta_deg)
        self.scan_processor = ScanProcessor()
        
        # Create and configure environment map
        self.environment_map = self._create_default_map()
        self.map_matcher = MapMatcher(self.environment_map)
        
        # Statistics
        self.scans_processed = 0
        self.matches_successful = 0
        self.matches_failed = 0
        
        logger.info("LocalizedRobot initialized")
    
    def _create_default_map(self) -> EnvironmentMap:
        """Create default warehouse environment map."""
        env_map = EnvironmentMap(width_mm=3000, height_mm=2000)
        
        # Add perimeter walls
        env_map.add_wall(0, 0, 3000, 0)          # Bottom
        env_map.add_wall(3000, 0, 3000, 2000)    # Right
        env_map.add_wall(3000, 2000, 0, 2000)    # Top
        env_map.add_wall(0, 2000, 0, 0)          # Left
        
        # Add internal walls/obstacles
        env_map.add_wall(1500, 600, 1500, 1400)  # Central column
        env_map.add_wall(500, 1000, 700, 1000)   # Obstacle
        
        # Add known landmarks
        env_map.add_landmark("ChargeStation", 100, 100)
        env_map.add_landmark("StorageA", 1500, 200)
        env_map.add_landmark("StorageB", 2800, 1800)
        env_map.add_landmark("Workspace", 1500, 1000)
        
        logger.info("Default environment map created")
        return env_map
    
    def start(self):
        """Start LiDAR acquisition."""
        logger.info("Starting LiDAR acquisition...")
        self.lidar.start()
        time.sleep(0.5)  # Let it stabilize
    
    def stop(self):
        """Stop LiDAR acquisition."""
        logger.info("Stopping LiDAR acquisition...")
        self.lidar.stop()
    
    def update_position(self, 
                       odom_x_mm: float, 
                       odom_y_mm: float, 
                       odom_theta_deg: float) -> dict:
        """
        Update robot position from odometry.
        
        Args:
            odom_x_mm: Odometry X in mm
            odom_y_mm: Odometry Y in mm
            odom_theta_deg: Odometry heading in degrees
        
        Returns:
            Updated pose information
        """
        # Update from odometry
        pose = self.pose_estimator.update_from_odom(odom_x_mm, odom_y_mm, odom_theta_deg)
        
        # Get current LiDAR scan
        raw_scan = self.lidar.get_raw_scan()
        
        if not raw_scan:
            return {
                "status": "no_scan",
                "pose": pose.to_dict()
            }
        
        # Process scan
        self.scans_processed += 1
        points = self.scan_processor.process_scan(raw_scan)
        
        if len(points) == 0:
            return {
                "status": "no_valid_points",
                "pose": pose.to_dict()
            }
        
        # Filter noise
        filtered_points = self.scan_processor.filter_noise(points)
        
        # Attempt map matching
        robot_pose_tuple = (pose.x_mm, pose.y_mm, pose.theta_deg)
        match_result = self.map_matcher.match_scan(
            filtered_points,
            robot_pose_tuple,
            search_radius_mm=150,
            angle_step_deg=5
        )
        
        # Apply correction if confident
        if match_result["confidence"] > 0.6:
            best_x, best_y, best_theta = match_result["best_match"]
            dx = best_x - pose.x_mm
            dy = best_y - pose.y_mm
            dtheta = best_theta - pose.theta_deg
            
            self.pose_estimator.update_from_scan_match(
                dx, dy, dtheta,
                confidence=match_result["confidence"]
            )
            self.matches_successful += 1
        else:
            self.matches_failed += 1
        
        # Detect walls and landmarks
        walls = self.scan_processor.detect_walls(filtered_points)
        landmarks = self.map_matcher.find_landmarks_in_scan(
            filtered_points, robot_pose_tuple
        )
        
        # Get updated pose
        final_pose = self.pose_estimator.get_pose()
        
        return {
            "status": "updated",
            "pose": final_pose.to_dict(),
            "scan_points": len(filtered_points),
            "walls_detected": len(walls),
            "landmarks_detected": landmarks,
            "match_confidence": match_result["confidence"],
            "match_status": "applied" if match_result["confidence"] > 0.6 else "rejected"
        }
    
    def get_position(self) -> dict:
        """Get current robot position."""
        pose = self.pose_estimator.get_pose()
        drift = self.pose_estimator.get_drift_estimate()
        
        return {
            "pose": pose.to_dict(),
            "drift": drift,
            "statistics": {
                "scans_processed": self.scans_processed,
                "matches_successful": self.matches_successful,
                "matches_failed": self.matches_failed,
                "success_rate": (self.matches_successful / max(1, self.scans_processed + self.matches_failed))
            }
        }
    
    def navigate_to_landmark(self, landmark_name: str) -> dict:
        """
        Navigate to a known landmark.
        
        Args:
            landmark_name: Name of the landmark
        
        Returns:
            Navigation information
        """
        if landmark_name not in self.environment_map.landmarks:
            return {"error": f"Landmark '{landmark_name}' not found"}
        
        land_x, land_y = self.environment_map.landmarks[landmark_name]
        distance = self.pose_estimator.get_distance_to_point(land_x, land_y)
        bearing = self.pose_estimator.get_bearing_to_point(land_x, land_y)
        
        return {
            "landmark": landmark_name,
            "position": {"x": land_x, "y": land_y},
            "distance_mm": distance,
            "bearing_deg": bearing,
            "current_pose": self.pose_estimator.get_pose_dict()
        }
    
    def reset_position(self, x_mm: float = 1500, y_mm: float = 1000, theta_deg: float = 0):
        """Reset robot position (e.g., when manually moved)."""
        self.pose_estimator.set_absolute_position(x_mm, y_mm, theta_deg)
        logger.info(f"Position reset to ({x_mm}, {y_mm}, {theta_deg}°)")


def example_continuous_localization():
    """Example: Continuous position tracking."""
    logger.info("")
    logger.info("═" * 60)
    logger.info("EXAMPLE: Continuous Localization")
    logger.info("═" * 60)
    logger.info("")
    
    robot = LocalizedRobot()
    
    try:
        # Start LiDAR
        robot.start()
        
        # Simulate odometry updates and localization
        logger.info("Robot position updates (press Ctrl+C to stop):")
        logger.info("")
        
        time_step = 0
        while time_step < 10:  # Run for 10 steps
            time_step += 1
            
            # Simulate odometry (robot moving forward)
            simulated_x = 1500 + time_step * 50
            simulated_y = 1000
            simulated_theta = 0
            
            # Update position with scan matching
            result = robot.update_position(simulated_x, simulated_y, simulated_theta)
            
            if result["status"] == "updated":
                pose = result["pose"]
                logger.info(f"Step {time_step}:")
                logger.info(f"  Position: ({pose['x_mm']:.0f}, {pose['y_mm']:.0f})mm")
                logger.info(f"  Heading: {pose['theta_deg']:.1f}°")
                logger.info(f"  Match: conf={result['match_confidence']:.2f}, "
                           f"status={result['match_status']}")
                logger.info(f"  Detected: {result['walls_detected']} walls, "
                           f"{len(result['landmarks_detected'])} landmarks")
            
            time.sleep(0.5)
        
        # Show final statistics
        logger.info("")
        position_info = robot.get_position()
        stats = position_info["statistics"]
        logger.info(f"Statistics:")
        logger.info(f"  Scans processed: {stats['scans_processed']}")
        logger.info(f"  Successful matches: {stats['matches_successful']}")
        logger.info(f"  Failed matches: {stats['matches_failed']}")
        logger.info(f"  Success rate: {stats['success_rate']*100:.1f}%")
        
        # Example navigation
        logger.info("")
        logger.info("Navigation example:")
        nav = robot.navigate_to_landmark("ChargeStation")
        logger.info(f"  Distance to ChargeStation: {nav['distance_mm']:.0f}mm")
        logger.info(f"  Bearing: {nav['bearing_deg']:.1f}°")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        robot.stop()
        logger.info("Example completed")


if __name__ == "__main__":
    example_continuous_localization()
