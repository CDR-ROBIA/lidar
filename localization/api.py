"""
Localization API Integration

Provides REST API endpoints for localization and positioning.
Can be integrated with the ROS FastAPI backend.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Tuple, Optional
import logging

from .pose_estimator import PoseEstimator, Pose
from .map_matcher import EnvironmentMap, MapMatcher
from .scan_processor import ScanProcessor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/localization", tags=["localization"])


# ============ Pydantic Models ============

class PoseRequest(BaseModel):
    x_mm: float
    y_mm: float
    theta_deg: float


class ScanData(BaseModel):
    """Raw LiDAR scan data."""
    measurements: List[Tuple[int, float, float]]  # (quality, angle, distance)


class OdometryUpdate(BaseModel):
    """Odometry position update."""
    x_mm: float
    y_mm: float
    theta_deg: float
    timestamp: float


class ScanMatchResult(BaseModel):
    """Scan matching result."""
    best_match: Tuple[float, float, float]
    confidence: float
    score: Optional[float] = None


class LocalizationState(BaseModel):
    """Current localization state."""
    pose_x_mm: float
    pose_y_mm: float
    pose_theta_deg: float
    confidence: float
    source: str
    drift_estimate: Dict


# ============ Global State ============

pose_estimator: Optional[PoseEstimator] = None
map_matcher: Optional[MapMatcher] = None
scan_processor: Optional[ScanProcessor] = None


def initialize_localization(init_x: float = 0.0, 
                           init_y: float = 0.0, 
                           init_theta: float = 0.0):
    """Initialize localization system."""
    global pose_estimator, map_matcher, scan_processor
    
    pose_estimator = PoseEstimator(init_x, init_y, init_theta)
    
    # Create default environment map
    env_map = EnvironmentMap(width_mm=3000.0, height_mm=2000.0)
    
    # Add some example walls (can be customized)
    env_map.add_wall(0, 0, 3000, 0)      # Bottom wall
    env_map.add_wall(3000, 0, 3000, 2000)  # Right wall
    env_map.add_wall(3000, 2000, 0, 2000)  # Top wall
    env_map.add_wall(0, 2000, 0, 0)      # Left wall
    
    map_matcher = MapMatcher(env_map)
    scan_processor = ScanProcessor()
    
    logger.info("Localization system initialized")


# ============ API Endpoints ============

@router.post("/init")
async def initialize(pose: PoseRequest):
    """Initialize localization system with starting pose."""
    try:
        initialize_localization(pose.x_mm, pose.y_mm, pose.pose_theta_deg)
        return {"status": "initialized", "pose": pose.model_dump()}
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/state")
async def get_localization_state() -> LocalizationState:
    """Get current localization state."""
    if pose_estimator is None:
        raise HTTPException(status_code=503, detail="Localization not initialized")
    
    pose = pose_estimator.get_pose()
    drift = pose_estimator.get_drift_estimate()
    
    return LocalizationState(
        pose_x_mm=pose.x_mm,
        pose_y_mm=pose.y_mm,
        pose_theta_deg=pose.theta_deg,
        confidence=pose.confidence,
        source=pose.source,
        drift_estimate=drift
    )


@router.post("/odom")
async def update_from_odometry(odom: OdometryUpdate) -> Dict:
    """Update pose from odometry data."""
    if pose_estimator is None:
        raise HTTPException(status_code=503, detail="Localization not initialized")
    
    try:
        pose = pose_estimator.update_from_odom(odom.x_mm, odom.y_mm, odom.theta_deg)
        return {
            "status": "updated",
            "pose": pose.to_dict()
        }
    except Exception as e:
        logger.error(f"Odometry update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scan")
async def process_and_match_scan(scan: ScanData) -> Dict:
    """
    Process LiDAR scan and perform map matching.
    
    Returns matching results and detected landmarks.
    """
    if pose_estimator is None or map_matcher is None or scan_processor is None:
        raise HTTPException(status_code=503, detail="Localization not initialized")
    
    try:
        # Process scan
        measurements = [(q, a, d) for q, a, d in scan.measurements]
        points = scan_processor.process_scan(measurements)
        
        if len(points) == 0:
            return {"status": "no_valid_points"}
        
        # Filter noise
        filtered_points = scan_processor.filter_noise(points)
        
        # Detect walls
        walls = scan_processor.detect_walls(filtered_points)
        
        # Get current pose
        current_pose = pose_estimator.get_pose()
        robot_pose = (current_pose.x_mm, current_pose.y_mm, current_pose.theta_deg)
        
        # Perform map matching
        match_result = map_matcher.match_scan(filtered_points, robot_pose)
        
        # Find landmarks
        detected_landmarks = map_matcher.find_landmarks_in_scan(
            filtered_points, 
            robot_pose
        )
        
        # Get scan statistics
        stats = scan_processor.get_statistics(measurements)
        
        return {
            "status": "processed",
            "points_count": len(filtered_points),
            "walls_detected": walls,
            "landmarks": detected_landmarks,
            "match": match_result,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Scan processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/set-absolute")
async def set_absolute_position(pose: PoseRequest) -> Dict:
    """
    Set absolute position (e.g., from external localization source).
    
    This resets odometry drift counters.
    """
    if pose_estimator is None:
        raise HTTPException(status_code=503, detail="Localization not initialized")
    
    try:
        pose_estimator.set_absolute_position(pose.x_mm, pose.y_mm, pose.theta_deg)
        current_pose = pose_estimator.get_pose()
        return {
            "status": "set",
            "pose": current_pose.to_dict()
        }
    except Exception as e:
        logger.error(f"Set absolute position error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/distance-to/{landmark_name}")
async def get_distance_to_landmark(landmark_name: str) -> Dict:
    """Get distance from robot to named landmark."""
    if pose_estimator is None or map_matcher is None:
        raise HTTPException(status_code=503, detail="Localization not initialized")
    
    if landmark_name not in map_matcher.map.landmarks:
        raise HTTPException(status_code=404, detail=f"Landmark '{landmark_name}' not found")
    
    land_x, land_y = map_matcher.map.landmarks[landmark_name]
    distance = pose_estimator.get_distance_to_point(land_x, land_y)
    bearing = pose_estimator.get_bearing_to_point(land_x, land_y)
    
    return {
        "landmark": landmark_name,
        "distance_mm": distance,
        "bearing_deg": bearing
    }


@router.get("/map")
async def get_environment_map() -> Dict:
    """Get current environment map."""
    if map_matcher is None:
        raise HTTPException(status_code=503, detail="Localization not initialized")
    
    return map_matcher.map.to_dict()


@router.post("/map/add-wall")
async def add_wall_to_map(x1: float, y1: float, x2: float, y2: float) -> Dict:
    """Add wall segment to environment map."""
    if map_matcher is None:
        raise HTTPException(status_code=503, detail="Localization not initialized")
    
    try:
        map_matcher.map.add_wall(x1, y1, x2, y2)
        return {"status": "wall_added", "from": (x1, y1), "to": (x2, y2)}
    except Exception as e:
        logger.error(f"Add wall error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/map/add-landmark")
async def add_landmark_to_map(name: str, x: float, y: float) -> Dict:
    """Add landmark to environment map."""
    if map_matcher is None:
        raise HTTPException(status_code=503, detail="Localization not initialized")
    
    try:
        map_matcher.map.add_landmark(name, x, y)
        return {"status": "landmark_added", "name": name, "position": (x, y)}
    except Exception as e:
        logger.error(f"Add landmark error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drift")
async def get_odometry_drift() -> Dict:
    """Get estimated odometry drift."""
    if pose_estimator is None:
        raise HTTPException(status_code=503, detail="Localization not initialized")
    
    return pose_estimator.get_drift_estimate()


@router.post("/drift/reset")
async def reset_drift() -> Dict:
    """Reset odometry drift counters."""
    if pose_estimator is None:
        raise HTTPException(status_code=503, detail="Localization not initialized")
    
    pose_estimator.reset_drift()
    return {"status": "drift_reset"}
