"""
Localization Demo

Demonstrates the localization system with sample data.
Shows how to use pose estimation, scan matching, and map navigation.
"""

from __future__ import annotations
import logging
import time
from localization.pose_estimator import PoseEstimator
from localization.map_matcher import EnvironmentMap, MapMatcher
from localization.scan_processor import ScanProcessor

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def demo_basic_positioning():
    """Demo 1: Basic robot positioning."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("DEMO 1: Basic Robot Positioning")
    logger.info("=" * 60)
    
    # Initialize pose estimator
    pose_est = PoseEstimator(init_x_mm=1500, init_y_mm=1000, init_theta_deg=0)
    
    logger.info(f"Initial pose: {pose_est.get_pose_dict()}")
    
    # Simulate odometry updates
    logger.info("Updating from odometry...")
    for i in range(3):
        # Simulated odometry reading (robot moving forward)
        new_x = 1500 + (i + 1) * 100
        new_y = 1000
        new_theta = 0
        
        pose = pose_est.update_from_odom(new_x, new_y, new_theta)
        logger.info(f"  Step {i+1}: x={pose.x_mm:.1f}, y={pose.y_mm:.1f}, "
                   f"θ={pose.theta_deg:.1f}°, conf={pose.confidence:.2f}")
    
    logger.info("")


def demo_scan_processing():
    """Demo 2: LiDAR scan processing."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("DEMO 2: LiDAR Scan Processing")
    logger.info("=" * 60)
    
    processor = ScanProcessor()
    
    # Create sample scan (simulated)
    sample_scan = [
        (15, 0.0, 500),      # Quality 15, angle 0°, distance 500mm
        (15, 10.0, 520),     # Quality 15, angle 10°, distance 520mm
        (15, 20.0, 540),     # Quality 15, angle 20°, distance 540mm
        (2, 30.0, 1500),     # Low quality point (will be filtered)
        (15, 40.0, 560),
        (15, 90.0, 600),     # Perpendicular direction
        (15, 180.0, 1000),   # Behind robot
    ]
    
    logger.info(f"Processing {len(sample_scan)} measurements...")
    
    # Process scan
    points = processor.process_scan(sample_scan)
    logger.info(f"Valid points: {len(points)}")
    for i, (x, y) in enumerate(points[:5]):
        logger.info(f"  Point {i}: ({x:.1f}, {y:.1f}) mm")
    
    # Get statistics
    stats = processor.get_statistics(sample_scan)
    logger.info(f"Scan statistics:")
    logger.info(f"  Total points: {stats['total_points']}")
    logger.info(f"  Valid points: {stats['valid_points']}")
    logger.info(f"  Avg distance: {stats['avg_distance_mm']:.1f} mm")
    logger.info(f"  Avg quality: {stats['avg_quality']:.1f}")
    
    # Detect walls
    walls = processor.detect_walls(points)
    logger.info(f"Walls detected: {len(walls)}")
    for wall in walls:
        logger.info(f"  Wall at ({wall['center_x']:.0f}, {wall['center_y']:.0f}), "
                   f"angle={wall['angle_deg']:.1f}°, length={wall['length_mm']:.0f}mm")
    
    logger.info("")


def demo_mapping():
    """Demo 3: Environment mapping."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("DEMO 3: Environment Mapping")
    logger.info("=" * 60)
    
    # Create environment map
    env_map = EnvironmentMap(width_mm=3000, height_mm=2000)
    
    # Add walls (warehouse perimeter)
    logger.info("Adding walls to map...")
    env_map.add_wall(0, 0, 3000, 0)         # Bottom wall
    env_map.add_wall(3000, 0, 3000, 2000)   # Right wall
    env_map.add_wall(3000, 2000, 0, 2000)   # Top wall
    env_map.add_wall(0, 2000, 0, 0)         # Left wall
    
    # Add landmarks
    logger.info("Adding landmarks...")
    env_map.add_landmark("ChargeStation", 100, 100)
    env_map.add_landmark("StorageA", 1500, 100)
    env_map.add_landmark("StorageB", 2800, 1900)
    
    logger.info(f"Map created: {len(env_map.walls)} walls, {len(env_map.landmarks)} landmarks")
    
    # Check occupancy
    logger.info(f"Occupancy at (1500, 1000) [robot center]: {env_map.get_occupancy_at(1500, 1000)}")
    logger.info(f"Occupancy at (0, 0) [wall corner]: {env_map.get_occupancy_at(0, 0)}")
    logger.info("")


def demo_scan_matching():
    """Demo 4: Scan matching against map."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("DEMO 4: Scan Matching")
    logger.info("=" * 60)
    
    # Setup
    env_map = EnvironmentMap(width_mm=3000, height_mm=2000)
    env_map.add_wall(0, 0, 3000, 0)
    env_map.add_wall(3000, 0, 3000, 2000)
    env_map.add_wall(3000, 2000, 0, 2000)
    env_map.add_wall(0, 2000, 0, 0)
    
    matcher = MapMatcher(env_map)
    processor = ScanProcessor()
    
    # Create scan at specific location
    sample_scan = [
        (15, i*6, 500 + i*10) for i in range(30)  # Simulated scan
    ]
    
    points = processor.process_scan(sample_scan)
    
    # Current pose estimate (maybe slightly off)
    robot_pose = (1450, 1050, 2)  # Estimated
    
    logger.info(f"Current pose estimate: ({robot_pose[0]}, {robot_pose[1]}, {robot_pose[2]}°)")
    logger.info("Performing scan matching...")
    
    match_result = matcher.match_scan(points, robot_pose, search_radius_mm=100, angle_step_deg=10)
    
    best_x, best_y, best_theta = match_result["best_match"]
    confidence = match_result["confidence"]
    
    logger.info(f"Best matching pose: ({best_x:.1f}, {best_y:.1f}, {best_theta:.1f}°)")
    logger.info(f"Match confidence: {confidence:.2f}")
    
    logger.info("")


def demo_navigation():
    """Demo 5: Navigation to landmarks."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("DEMO 5: Navigation")
    logger.info("=" * 60)
    
    # Setup
    pose_est = PoseEstimator(init_x_mm=1500, init_y_mm=1000, init_theta_deg=0)
    env_map = EnvironmentMap(width_mm=3000, height_mm=2000)
    
    # Add landmarks
    env_map.add_landmark("ChargeStation", 100, 100)
    env_map.add_landmark("StorageA", 1500, 100)
    
    logger.info(f"Robot at: {pose_est.get_position_mm()}")
    logger.info(f"Robot heading: {pose_est.get_orientation_deg()}°")
    
    for landmark_name, (land_x, land_y) in env_map.landmarks.items():
        distance = pose_est.get_distance_to_point(land_x, land_y)
        bearing = pose_est.get_bearing_to_point(land_x, land_y)
        
        logger.info(f"To {landmark_name}: distance={distance:.0f}mm, bearing={bearing:.1f}°")
    
    logger.info("")


def demo_drift_tracking():
    """Demo 6: Odometry drift tracking."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("DEMO 6: Odometry Drift Tracking")
    logger.info("=" * 60)
    
    pose_est = PoseEstimator(init_x_mm=1500, init_y_mm=1000, init_theta_deg=0)
    
    logger.info("Simulating odometry with drift...")
    
    # Simulate robot moving in a circle
    for step in range(6):
        angle = step * 60  # 0°, 60°, 120°, 180°, 240°, 300°
        angle_rad = (angle * 3.14159) / 180
        
        # Position on circle
        x = 1500 + 500 * np.cos(angle_rad)
        y = 1000 + 500 * np.sin(angle_rad)
        
        pose_est.update_from_odom(x, y, float(angle))
        
        drift = pose_est.get_drift_estimate()
        logger.info(f"Step {step+1}: Position error={drift['total_drift_mm']:.0f}mm")
    
    logger.info("")


# Only import numpy if needed for demo
import numpy as np


def main():
    """Run all demos."""
    logger.info("")
    logger.info("╔═══════════════════════════════════════════════════════╗")
    logger.info("║  LOCALIZATION SYSTEM DEMONSTRATION                   ║")
    logger.info("╚═══════════════════════════════════════════════════════╝")
    
    try:
        demo_basic_positioning()
        demo_scan_processing()
        demo_mapping()
        demo_scan_matching()
        demo_navigation()
        demo_drift_tracking()
        
        logger.info("")
        logger.info("╔═══════════════════════════════════════════════════════╗")
        logger.info("║  ✓ All demos completed successfully                  ║")
        logger.info("╚═══════════════════════════════════════════════════════╝")
        logger.info("")
        
    except Exception as e:
        logger.error(f"Demo error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
