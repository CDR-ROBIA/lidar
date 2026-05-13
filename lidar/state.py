"""
State dataclass for LiDAR robot localization.

Encapsulates all robot state fields with full type annotations.
Thread-safe reading through immutable copying.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class RobotState:
    """
    Immutable snapshot of robot state at a point in time.
    
    All numeric fields use specific units:
    - x, y coordinates in millimeters
    - heading in radians (absolute orientation in trigonometric frame)
    - distances in millimeters
    - angles in degrees (LIDAR frame)
    """
    
    robot_x: float = 0.0
    """X coordinate of robot position (mm) in absolute frame."""
    
    robot_y: float = 0.0
    """Y coordinate of robot position (mm) in absolute frame."""
    
    heading: float = 0.0
    """Robot orientation in radians (absolute, trigonometric)."""
    
    scan: List = field(default_factory=list)
    """Raw LiDAR scan data (list of tuples from RPLidar)."""
    
    dist_p1_mm: Optional[float] = None
    """Measured distance to pole P1 (mm), or None if not detected."""
    
    dist_p2_mm: Optional[float] = None
    """Measured distance to pole P2 (mm), or None if not detected."""
    
    dist_p3_mm: Optional[float] = None
    """Measured distance to pole P3 (mm), or None if not detected."""
    
    angle_p1_deg: float = 0.0
    """Measured angle to pole P1 (degrees, LIDAR frame)."""
    
    angle_p2_deg: float = 0.0
    """Measured angle to pole P2 (degrees, LIDAR frame)."""
    
    angle_p3_deg: float = 0.0
    """Measured angle to pole P3 (degrees, LIDAR frame)."""
    
    num_scan: int = 0
    """Scan counter (monotonically increasing)."""
    
    status: str = "startup..."
    """Status message (OK, missing pole, trilateration failed, etc)."""
    
    def is_valid(self) -> bool:
        """
        Check if current state represents a valid position.
        
        Returns:
            bool: True if position is in bounds and status is "OK".
        """
        return self.status == "OK" and self._in_bounds()
    
    def _in_bounds(self) -> bool:
        """Check if robot position is within arena bounds."""
        return 0 <= self.robot_x <= 3000 and 0 <= self.robot_y <= 2000
    
    def get_position(self) -> Tuple[float, float, float]:
        """
        Get position as (x_mm, y_mm, heading_rad).
        
        Returns:
            Tuple[float, float, float]: Robot position and orientation.
        """
        return (self.robot_x, self.robot_y, self.heading)
