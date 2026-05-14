#!/usr/bin/env python3
"""
Quick Test Script

Runs basic tests to verify localization system functionality.
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)-8s] %(message)s"
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing imports...")
    try:
        from localization.scan_processor import ScanProcessor
        from localization.pose_estimator import PoseEstimator, Pose
        from localization.map_matcher import EnvironmentMap, MapMatcher
        from localization.api import router
        logger.info("✓ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_scan_processor():
    """Test scan processor."""
    logger.info("Testing ScanProcessor...")
    try:
        from localization.scan_processor import ScanProcessor
        
        processor = ScanProcessor()
        
        # Create sample scan
        scan = [
            (15, 0.0, 500),
            (15, 10.0, 520),
            (15, 20.0, 540),
            (15, 90.0, 600),
            (15, 180.0, 1000),
        ]
        
        points = processor.process_scan(scan)
        assert len(points) == 5, "Expected 5 valid points"
        
        stats = processor.get_statistics(scan)
        assert stats["valid_points"] == 5, "Expected 5 valid points in stats"
        
        logger.info(f"✓ ScanProcessor OK ({len(points)} points processed)")
        return True
    except Exception as e:
        logger.error(f"✗ ScanProcessor test failed: {e}")
        return False


def test_pose_estimator():
    """Test pose estimator."""
    logger.info("Testing PoseEstimator...")
    try:
        from localization.pose_estimator import PoseEstimator
        
        pose_est = PoseEstimator(1500, 1000, 0)
        
        # Test odometry update
        pose = pose_est.update_from_odom(1550, 1020, 5)
        assert pose.x_mm == 1550, "X position mismatch"
        assert pose.y_mm == 1020, "Y position mismatch"
        
        # Test absolute position
        pose_est.set_absolute_position(2000, 500, 90)
        pose = pose_est.get_pose()
        assert pose.x_mm == 2000, "X position after set failed"
        assert pose.confidence > 0.9, "Confidence after set too low"
        
        # Test distance calculation
        dist = pose_est.get_distance_to_point(2000, 600)
        assert abs(dist - 100.0) < 1, "Distance calculation error"
        
        logger.info("✓ PoseEstimator OK")
        return True
    except Exception as e:
        logger.error(f"✗ PoseEstimator test failed: {e}")
        return False


def test_environment_map():
    """Test environment map."""
    logger.info("Testing EnvironmentMap...")
    try:
        from localization.map_matcher import EnvironmentMap
        
        env_map = EnvironmentMap()
        
        # Add walls
        env_map.add_wall(0, 0, 1000, 0)
        assert len(env_map.walls) == 1, "Wall not added"
        
        # Add landmarks
        env_map.add_landmark("Test", 500, 500)
        assert "Test" in env_map.landmarks, "Landmark not added"
        assert env_map.landmarks["Test"] == (500, 500), "Landmark position wrong"
        
        # Check occupancy
        occupancy = env_map.get_occupancy_at(500, 500)
        assert occupancy in [0, 1], "Invalid occupancy value"
        
        logger.info("✓ EnvironmentMap OK")
        return True
    except Exception as e:
        logger.error(f"✗ EnvironmentMap test failed: {e}")
        return False


def test_map_matcher():
    """Test map matcher."""
    logger.info("Testing MapMatcher...")
    try:
        from localization.map_matcher import EnvironmentMap, MapMatcher
        import numpy as np
        
        env_map = EnvironmentMap()
        env_map.add_wall(0, 0, 1000, 0)
        
        matcher = MapMatcher(env_map)
        
        # Create sample points
        points = np.array([[100, 0], [200, 0], [300, 0]], dtype=np.float32)
        robot_pose = (500, 500, 0)
        
        result = matcher.match_scan(points, robot_pose)
        assert "best_match" in result, "No best_match in result"
        assert "confidence" in result, "No confidence in result"
        assert 0 <= result["confidence"] <= 1, "Confidence out of range"
        
        logger.info("✓ MapMatcher OK")
        return True
    except Exception as e:
        logger.error(f"✗ MapMatcher test failed: {e}")
        return False


def test_api_models():
    """Test Pydantic models."""
    logger.info("Testing API models...")
    try:
        from localization.api import (
            PoseRequest, OdometryUpdate, ScanData, LocalizationState
        )
        
        # Test PoseRequest
        pose_req = PoseRequest(x_mm=1500, y_mm=1000, theta_deg=0)
        assert pose_req.x_mm == 1500, "PoseRequest creation failed"
        
        # Test ScanData
        scan_data = ScanData(measurements=[(15, 0.0, 500), (15, 10.0, 520)])
        assert len(scan_data.measurements) == 2, "ScanData creation failed"
        
        logger.info("✓ API models OK")
        return True
    except Exception as e:
        logger.error(f"✗ API models test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    logger.info("")
    logger.info("╔═══════════════════════════════════════╗")
    logger.info("║     LOCALIZATION SYSTEM TEST SUITE    ║")
    logger.info("╚═══════════════════════════════════════╝")
    logger.info("")
    
    tests = [
        ("Imports", test_imports),
        ("ScanProcessor", test_scan_processor),
        ("PoseEstimator", test_pose_estimator),
        ("EnvironmentMap", test_environment_map),
        ("MapMatcher", test_map_matcher),
        ("API Models", test_api_models),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"✗ {name} test crashed: {e}")
            failed += 1
    
    logger.info("")
    logger.info("╔═══════════════════════════════════════╗")
    logger.info(f"║ Results: {passed} passed, {failed} failed       ║")
    logger.info("╚═══════════════════════════════════════╝")
    logger.info("")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
