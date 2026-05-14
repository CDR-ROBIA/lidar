"""
FastAPI Integration Guide

Shows how to integrate the localization system into the ROS FastAPI backend.
Add this to your backend/app.py or create a separate router.
"""

# Example integration for FastAPI backend

# ============ In backend/app.py ============

"""
from localization.api import router as localization_router
from localization.api import initialize_localization

# In your lifespan startup:
@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting ROS Robot Control Platform")
    
    # Start ROS client
    ros_client.connect()
    telemetry_buffer.start_background_updates()
    
    # Initialize localization system
    initialize_localization(
        init_x=1500,
        init_y=1000,
        init_theta=0
    )
    
    yield
    
    # Cleanup
    telemetry_buffer.stop_background_updates()
    ros_client.close()
    logger.info("Application stopped")

# Include localization routes
app.include_router(localization_router)
"""


# ============ Integration with ROS Topics ============

"""
from localization.api import pose_estimator, scan_processor, map_matcher
from localization.pose_estimator import PoseEstimator
from ros.ros_client import ros_client
import roslibpy

class LocalizationSubscriber:
    \"\"\"Subscribe to ROS topics and update localization.\"\"\"
    
    def __init__(self):
        self.odom_topic = None
        self.scan_topic = None
    
    def start(self, ros_client: roslibpy.Ros):
        \"\"\"Start subscribing to ROS topics.\"\"\"
        # Subscribe to odometry
        self.odom_topic = roslibpy.Topic(
            ros_client, '/odom', 'nav_msgs/Odometry'
        )
        self.odom_topic.subscribe(self._on_odom)
        
        # Subscribe to LiDAR scan
        self.scan_topic = roslibpy.Topic(
            ros_client, '/scan', 'sensor_msgs/LaserScan'
        )
        self.scan_topic.subscribe(self._on_scan)
        
        logger.info("Localization subscribers started")
    
    def _on_odom(self, message: dict):
        \"\"\"Callback for odometry topic.\"\"\"
        if pose_estimator is None:
            return
        
        # Extract pose from odometry
        pose = message.get('pose', {}).get('pose', {})
        position = pose.get('position', {})
        orientation = pose.get('orientation', {})
        
        x_m = float(position.get('x', 0))
        y_m = float(position.get('y', 0))
        
        # Convert from radians to degrees
        from math import atan2, degrees
        q_z = float(orientation.get('z', 0))
        q_w = float(orientation.get('w', 1))
        theta_deg = degrees(2 * atan2(q_z, q_w))
        
        # Update localization (convert meters to mm)
        pose_estimator.update_from_odom(
            x_m * 1000,
            y_m * 1000,
            theta_deg
        )
    
    def _on_scan(self, message: dict):
        \"\"\"Callback for LiDAR scan topic.\"\"\"
        if scan_processor is None or map_matcher is None:
            return
        
        # Extract scan data
        ranges = message.get('ranges', [])
        angle_min = float(message.get('angle_min', 0))
        angle_increment = float(message.get('angle_increment', 0))
        
        # Convert to RPLidar format (quality, angle, distance)
        scan_data = []
        for i, distance_m in enumerate(ranges):
            if distance_m > 0:
                angle_deg = (angle_min + i * angle_increment) * 180 / 3.14159
                quality = 15  # ROS scan has no quality, use default
                distance_mm = distance_m * 1000
                scan_data.append((quality, angle_deg, distance_mm))
        
        if not scan_data:
            return
        
        # Process scan
        points = scan_processor.process_scan(scan_data)
        if len(points) == 0:
            return
        
        # Perform map matching
        current_pose = pose_estimator.get_pose()
        robot_pose = (
            current_pose.x_mm,
            current_pose.y_mm,
            current_pose.theta_deg
        )
        
        match_result = map_matcher.match_scan(
            points, robot_pose,
            search_radius_mm=150,
            angle_step_deg=5
        )
        
        # Apply correction if confident
        if match_result['confidence'] > 0.6:
            best_x, best_y, best_theta = match_result['best_match']
            dx = best_x - current_pose.x_mm
            dy = best_y - current_pose.y_mm
            dtheta = best_theta - current_pose.theta_deg
            
            pose_estimator.update_from_scan_match(
                dx, dy, dtheta,
                confidence=match_result['confidence']
            )


# In your app startup:
localization_sub = LocalizationSubscriber()

@app.on_event('startup')
async def startup():
    # ... other startup code ...
    localization_sub.start(ros_client.client)
"""


# ============ WebSocket Extension ============

"""
from fastapi import WebSocketException
import asyncio

@app.websocket(\"/ws/localization\")
async def websocket_localization(websocket: WebSocket):
    \"\"\"WebSocket for real-time localization updates.\"\"\"
    await websocket.accept()
    
    try:
        while True:
            # Send current localization state
            if pose_estimator is not None:
                pose = pose_estimator.get_pose()
                drift = pose_estimator.get_drift_estimate()
                
                message = {
                    \"type\": \"pose_update\",
                    \"pose\": pose.to_dict(),
                    \"drift\": drift
                }
                
                await websocket.send_json(message)
            
            # Update frequency (10 Hz)
            await asyncio.sleep(0.1)
    
    except WebSocketDisconnect:
        logger.info(\"Localization WebSocket disconnected\")
"""


# ============ Dashboard Integration ============

"""
// In frontend/app.js

// WebSocket connection for localization
const locWebSocket = new WebSocket('ws://localhost:8000/ws/localization');

locWebSocket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'pose_update') {
        // Update robot position on canvas
        const pose = data.pose;
        robotX = pose.x_mm;
        robotY = pose.y_mm;
        robotTheta = pose.theta_deg;
        
        // Update info display
        document.getElementById('position').textContent = 
            `Position: (${Math.round(robotX)}, ${Math.round(robotY)})mm`;
        document.getElementById('heading').textContent =
            `Heading: ${pose.theta_deg.toFixed(1)}°`;
        document.getElementById('confidence').textContent =
            `Confidence: ${(pose.confidence * 100).toFixed(1)}%`;
    }
};

// API calls for navigation
async function navigateToLandmark(landmarkName) {
    const response = await fetch(
        `/api/localization/distance-to/${landmarkName}`
    );
    const data = await response.json();
    
    console.log(`Distance to ${landmarkName}: ${data.distance_mm.toFixed(0)}mm`);
    console.log(`Bearing: ${data.bearing_deg.toFixed(1)}°`);
    
    // Update UI or trigger movement commands
}

// Initialize localization
async function initLocalization() {
    const response = await fetch('/api/localization/init', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            x_mm: 1500,
            y_mm: 1000,
            theta_deg: 0
        })
    });
    
    if (response.ok) {
        console.log('Localization initialized');
    }
}

// Add landmark to map
async function addLandmark(name, x, y) {
    const response = await fetch(
        `/api/localization/map/add-landmark?name=${name}&x=${x}&y=${y}`,
        {method: 'POST'}
    );
    
    if (response.ok) {
        console.log(`Landmark ${name} added at (${x}, ${y})`);
    }
}
"""


# ============ Configuration ============

"""
# In backend/.env or config.py

# Localization settings
LOCALIZATION_ENABLED=true
LOCALIZATION_INIT_X_MM=1500
LOCALIZATION_INIT_Y_MM=1000
LOCALIZATION_INIT_THETA_DEG=0

# Scan processor
SCAN_MAX_RANGE_MM=6000
SCAN_MIN_RANGE_MM=50
SCAN_QUALITY_THRESHOLD=5

# Map matcher
MAP_SEARCH_RADIUS_MM=150
MAP_ANGLE_STEP_DEG=5
MAP_MATCH_CONFIDENCE_THRESHOLD=0.6

# Pose estimator
POSE_DRIFT_THRESHOLD_MM=1000
"""


# ============ Docker Compose Update ============

"""
# Add to docker-compose.yml if using container

services:
  backend:
    # ... existing config ...
    environment:
      - LOCALIZATION_ENABLED=true
      - LOCALIZATION_INIT_X_MM=1500
      - LOCALIZATION_INIT_Y_MM=1000
      - SCAN_MAX_RANGE_MM=6000
      # ... other env vars ...
"""


# ============ Monitoring and Logging ============

"""
# Add monitoring endpoint

@router.get('/localization/health')
async def localization_health():
    \"\"\"Check localization system health.\"\"\"
    if pose_estimator is None:
        return {
            'status': 'not_initialized',
            'ready': False
        }
    
    drift = pose_estimator.get_drift_estimate()
    
    return {
        'status': 'healthy',
        'ready': True,
        'confidence': pose_estimator.pose.confidence,
        'drift_mm': drift['total_drift_mm'],
        'scans_since_reset': drift['scan_match_confidence'],
        'source': pose_estimator.pose.source
    }


# Add diagnostic endpoint

@router.get('/localization/diagnostics')
async def localization_diagnostics():
    \"\"\"Get detailed diagnostics about localization.\"\"\"
    if pose_estimator is None:
        return {'error': 'not_initialized'}
    
    pose = pose_estimator.get_pose()
    drift = pose_estimator.get_drift_estimate()
    
    return {
        'pose': pose.to_dict(),
        'drift': drift,
        'landmarks': list(map_matcher.map.landmarks.keys()),
        'walls': len(map_matcher.map.walls),
        'last_match_result': map_matcher.last_match_result,
        'timestamp': time.time()
    }
"""

# ============ Key Integration Points ============

"""
1. INITIALIZATION
   - Call initialize_localization() in app startup
   - Set initial robot pose based on known position

2. ODOMETRY UPDATES
   - Subscribe to ROS /odom topic
   - Call pose_estimator.update_from_odom()
   - Update frequency: 10-20 Hz

3. LIDAR INTEGRATION  
   - Subscribe to ROS /scan topic
   - Call scan_processor.process_scan()
   - Call map_matcher.match_scan()
   - Apply correction with update_from_scan_match()
   - Update frequency: 5-10 Hz

4. API ENDPOINTS
   - GET /api/localization/state - Current pose
   - POST /api/localization/odom - Update from odometry
   - POST /api/localization/scan - Process scan
   - GET /api/localization/distance-to/{landmark} - Navigation
   - GET /api/localization/map - Get environment map
   - POST /api/localization/map/* - Modify map

5. WEBSOCKET
   - Connect to /ws/localization
   - Receive pose updates in real-time
   - Push to frontend for visualization

6. MONITORING
   - Check /api/localization/health
   - Monitor /api/localization/diagnostics
   - Log drift accumulation
   - Alert on confidence drops
"""
