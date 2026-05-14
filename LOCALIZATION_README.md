# Robot Localization System

Complete Python-based localization and positioning module for robot navigation in known environments.

## Overview

This localization system provides:

1. **Pose Estimation** - Tracks robot position and orientation
2. **Scan Processing** - Converts LiDAR data to point clouds
3. **Map Matching** - Aligns scans to known environment maps
4. **Landmark Detection** - Identifies and uses known features
5. **Drift Tracking** - Monitors odometry accuracy

**Note:** This system handles **positioning and localization only**. Robot movement commands are handled separately by the main control system.

---

## Architecture

```
┌─────────────────────────────────────────┐
│      LiDAR Raw Data                     │
│   (quality, angle, distance)            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│    ScanProcessor                        │
│  - Polar→Cartesian conversion          │
│  - Quality filtering                   │
│  - Noise filtering                     │
│  - Wall detection                      │
└──────────────┬──────────────────────────┘
               │
               ├─────────────┬────────────┐
               ▼             ▼            ▼
          Point Cloud   Wall Data    Statistics
               │
               ▼
┌─────────────────────────────────────────┐
│    MapMatcher                           │
│  - Grid search matching                │
│  - Confidence scoring                  │
│  - Landmark detection                  │
└──────────────┬──────────────────────────┘
               │
               ▼
        Pose Correction
               │
               ▼
┌─────────────────────────────────────────┐
│    PoseEstimator                        │
│  - Pose tracking                       │
│  - Sensor fusion                       │
│  - Drift estimation                    │
└──────────────┬──────────────────────────┘
               │
               ▼
        Robot Pose (x, y, θ)
```

---

## Components

### 1. ScanProcessor

Converts raw LiDAR measurements to usable point clouds.

**Key Methods:**
- `process_scan(scan)` - Convert polar to Cartesian coordinates
- `filter_noise(points)` - Remove outliers
- `detect_walls(points)` - Find wall segments
- `get_statistics(scan)` - Scan quality metrics

**Usage:**
```python
from localization.scan_processor import ScanProcessor

processor = ScanProcessor(
    max_range_mm=6000,
    min_range_mm=50,
    quality_threshold=5
)

# Process raw LiDAR data
scan = [(15, 0.0, 500), (15, 10.0, 520), ...]  # (quality, angle, distance)
points = processor.process_scan(scan)  # Returns (N, 2) point cloud

# Filter outliers
clean_points = processor.filter_noise(points)

# Detect walls
walls = processor.detect_walls(clean_points)
```

### 2. PoseEstimator

Maintains and updates robot pose estimate.

**Key Methods:**
- `update_from_odom(x, y, theta)` - Update from odometry
- `update_from_scan_match(dx, dy, dtheta, confidence)` - Apply scan matching correction
- `set_absolute_position(x, y, theta)` - Set absolute pose
- `get_distance_to_point(x, y)` - Distance to target
- `get_bearing_to_point(x, y)` - Heading to target
- `get_drift_estimate()` - Odometry error tracking

**Usage:**
```python
from localization.pose_estimator import PoseEstimator

pose_est = PoseEstimator(init_x_mm=1500, init_y_mm=1000, init_theta_deg=0)

# Update from odometry
pose = pose_est.update_from_odom(1550, 1020, 5.0)
print(f"Pose: {pose.x_mm}, {pose.y_mm}, {pose.theta_deg}°")

# Apply scan matching correction
pose = pose_est.update_from_scan_match(dx_mm=-10, dy_mm=5, dtheta_deg=-1, confidence=0.8)

# Get distance to target
target_dist = pose_est.get_distance_to_point(500, 500)
```

### 3. MapMatcher

Matches LiDAR scans against known environment maps.

**Features:**
- Occupancy grid representation
- Wall and landmark storage
- Grid search matching
- Confidence scoring

**Usage:**
```python
from localization.map_matcher import EnvironmentMap, MapMatcher

# Create environment map
env_map = EnvironmentMap(width_mm=3000, height_mm=2000)

# Add known walls
env_map.add_wall(0, 0, 3000, 0)        # Bottom wall
env_map.add_wall(3000, 0, 3000, 2000)  # Right wall

# Add landmarks
env_map.add_landmark("ChargeStation", 100, 100)

# Create matcher
matcher = MapMatcher(env_map)

# Match scan
robot_pose = (1500, 1000, 0)
result = matcher.match_scan(
    scan_points=points,
    robot_pose=robot_pose,
    search_radius_mm=200,
    angle_step_deg=5
)

print(f"Best match: {result['best_match']}")
print(f"Confidence: {result['confidence']}")
```

---

## API Integration

The module includes a FastAPI integration (`localization/api.py`) with these endpoints:

### Initialization
```
POST /localization/init
Body: {"x_mm": 1500, "y_mm": 1000, "theta_deg": 0}
```

### Get State
```
GET /localization/state
Response: {
    "pose_x_mm": 1500,
    "pose_y_mm": 1000,
    "pose_theta_deg": 0,
    "confidence": 0.95,
    "source": "fused",
    "drift_estimate": {...}
}
```

### Update from Odometry
```
POST /localization/odom
Body: {"x_mm": 1550, "y_mm": 1020, "theta_deg": 5, "timestamp": 1234567890}
```

### Process and Match Scan
```
POST /localization/scan
Body: {
    "measurements": [
        [15, 0.0, 500],
        [15, 10.0, 520],
        ...
    ]
}
Response: {
    "status": "processed",
    "points_count": 120,
    "walls_detected": [...],
    "landmarks": {...},
    "match": {...},
    "statistics": {...}
}
```

### Distance to Landmark
```
GET /localization/distance-to/ChargeStation
Response: {
    "landmark": "ChargeStation",
    "distance_mm": 2100.5,
    "bearing_deg": 45.3
}
```

### Map Management
```
POST /localization/map/add-wall?x1=0&y1=0&x2=3000&y2=0
POST /localization/map/add-landmark?name=StorageA&x=1500&y=100
GET /localization/map
```

### Drift Management
```
GET /localization/drift
POST /localization/drift/reset
```

---

## Data Models

### Pose
```python
@dataclass
class Pose:
    x_mm: float           # X position in millimeters
    y_mm: float           # Y position in millimeters
    theta_deg: float      # Rotation in degrees (0-360)
    timestamp: float      # Unix timestamp
    confidence: float     # Confidence (0-1)
    source: str          # "init", "odom", "lidar", "fused", "absolute"
```

### Scan Statistics
```python
{
    "total_points": 120,
    "valid_points": 118,
    "avg_distance_mm": 750.5,
    "min_distance_mm": 50.0,
    "max_distance_mm": 6000.0,
    "avg_quality": 14.2,
    "angle_coverage": 358.5  # degrees
}
```

### Wall Detection
```python
{
    "center_x": 1500.0,      # Center position
    "center_y": 100.0,
    "angle_deg": 0.0,        # Wall orientation
    "length_mm": 3000.0,
    "points_count": 45
}
```

---

## Example Usage

### Standalone Usage

```python
from localization.pose_estimator import PoseEstimator
from localization.scan_processor import ScanProcessor
from localization.map_matcher import EnvironmentMap, MapMatcher

# Initialize
pose_est = PoseEstimator(1500, 1000, 0)
processor = ScanProcessor()
env_map = EnvironmentMap()
matcher = MapMatcher(env_map)

# Main loop
while robot_running:
    # Get LiDAR data
    scan = lidar.get_raw_scan()
    
    # Process scan
    points = processor.process_scan(scan)
    
    # Get odometry update
    odom_x, odom_y, odom_theta = get_odometry()
    pose_est.update_from_odom(odom_x, odom_y, odom_theta)
    
    # Match to map
    current_pose = pose_est.get_pose()
    match = matcher.match_scan(
        points,
        (current_pose.x_mm, current_pose.y_mm, current_pose.theta_deg)
    )
    
    # Apply correction
    if match["confidence"] > 0.5:
        correction = calculate_correction(match)
        pose_est.update_from_scan_match(*correction)
    
    # Get current position
    x, y = pose_est.get_position_mm()
    print(f"Robot at: ({x:.0f}, {y:.0f})")
```

### With ROS Integration

```python
from localization import api as localization_api
from fastapi import FastAPI

app = FastAPI()

# Initialize localization
localization_api.initialize_localization(1500, 1000, 0)

# Include routes
app.include_router(localization_api.router)

# In your ROS subscriber callback:
@ros_subscriber('/odom')
def on_odom(msg):
    # Update localization with odom data
    localization_api.pose_estimator.update_from_odom(
        msg.pose.pose.position.x * 1000,  # Convert to mm
        msg.pose.pose.position.y * 1000,
        get_euler_z(msg.pose.pose.orientation)
    )
```

---

## Running the Demo

```bash
cd /home/ben/Downloads/lidar-main

# Run the demonstration
python -m localization.demo

# Or with logging
python -m localization.demo 2>&1 | tee demo_output.log
```

The demo shows:
1. Basic positioning and odometry updates
2. LiDAR scan processing and statistics
3. Environment mapping
4. Scan matching to known maps
5. Navigation to landmarks
6. Odometry drift tracking

---

## Configuration

### Environment Variables

```bash
# Scan processor
SCAN_MAX_RANGE_MM=6000
SCAN_MIN_RANGE_MM=50
SCAN_QUALITY_THRESHOLD=5

# Map matcher
MAP_SEARCH_RADIUS_MM=200
MAP_ANGLE_STEP_DEG=5

# Pose estimator
POSE_CONFIDENCE_THRESHOLD=0.5
```

### Customization

```python
# Custom scan processor
processor = ScanProcessor(
    max_range_mm=8000,
    min_range_mm=30,
    quality_threshold=10
)

# Custom environment map
env_map = EnvironmentMap(
    width_mm=5000,
    height_mm=3000,
    grid_cell_mm=100
)

# Custom map matcher
matcher = MapMatcher(env_map)
match = matcher.match_scan(
    scan_points,
    robot_pose,
    search_radius_mm=500,
    angle_step_deg=2.0
)
```

---

## Performance Notes

- **Scan Processing**: ~1-2ms per scan
- **Wall Detection**: ~5-10ms for 100-point cloud
- **Map Matching**: ~50-100ms (depends on search radius)
- **Memory**: ~10-20MB for full system with map

---

## Integration with lidar-main Project

This localization module is designed to work with the existing LiDAR acquisition system in `lidar-main`:

```python
from lidar.lidar import Lidar, find_lidar_port
from localization import pose_estimator, scan_processor, map_matcher

with Lidar(port=find_lidar_port()) as lidar:
    lidar.start()
    
    for scan in lidar.get_raw_scans():
        # Process with localization system
        points = scan_processor.process_scan(scan)
        match = map_matcher.match_scan(points, pose_estimator.get_pose()...)
        
        # Update position
        pose_estimator.update_from_scan_match(...)
```

---

## Troubleshooting

**Issue:** Low match confidence
- Ensure environment map walls match actual walls
- Increase search radius
- Check LiDAR data quality

**Issue:** Drift accumulating quickly
- Reduce odometry confidence weighting
- Increase scan matching frequency
- Check for moving obstacles

**Issue:** Landmarks not detected
- Add landmarks to map first
- Ensure landmark is within scan range
- Check landmark position accuracy

---

## Future Enhancements

- [ ] ICP (Iterative Closest Point) matching for higher accuracy
- [ ] Multi-hypothesis tracking for ambiguous environments
- [ ] Particle filter for non-linear motion models
- [ ] Loop closure detection
- [ ] Dynamic obstacle handling
- [ ] Global optimization using pose graph SLAM
