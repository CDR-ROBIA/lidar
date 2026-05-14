"""
Pure functions for localization calculations.

Trilateration and heading estimation algorithms.
No class, no state — pure mathematical functions.
"""

from typing import Optional, Tuple
import numpy as np


def trilaterate(
    pole_p1: np.ndarray,
    dist_p1_mm: float,
    pole_p2: np.ndarray,
    dist_p2_mm: float,
    pole_p3: np.ndarray,
    dist_p3_mm: float,
) -> Optional[np.ndarray]:
    """
    Solve trilateration problem using 3 distance measurements.
    
    Given three poles at known positions and measured distances,
    solve the 2x2 linear system derived from circle equations.
    
    The system is obtained by subtracting the circle equation of P1
    from those of P2 and P3, eliminating x² and y² terms.
    
    Args:
        pole_p1: Position of pole 1 as [x, y] (mm).
        dist_p1_mm: Measured distance to pole 1 (mm).
        pole_p2: Position of pole 2 as [x, y] (mm).
        dist_p2_mm: Measured distance to pole 2 (mm).
        pole_p3: Position of pole 3 as [x, y] (mm).
        dist_p3_mm: Measured distance to pole 3 (mm).
    
    Returns:
        np.ndarray: Position [x, y] in mm, or None if system is singular
                    (poles nearly collinear).
    
    Raises:
        ValueError: If any distance is invalid (≤0, ≥DIST_MAX, or None).
    """
    # Validate distances
    DIST_MAX = int(np.sqrt(3000**2 + 2000**2))  # ~3606 mm
    
    if (dist_p1_mm is None or dist_p2_mm is None or dist_p3_mm is None or
        dist_p1_mm <= 0 or dist_p2_mm <= 0 or dist_p3_mm <= 0 or
        dist_p1_mm > DIST_MAX or dist_p2_mm > DIST_MAX or dist_p3_mm > DIST_MAX):
        raise ValueError(f"Invalid distances: r1={dist_p1_mm}, r2={dist_p2_mm}, r3={dist_p3_mm}")
    
    # Build system: A * [x, y]^T = B
    # Derived from: (x - p1[0])² + (y - p1[1])² = r1²
    #               (x - p2[0])² + (y - p2[1])² = r2²
    #               (x - p3[0])² + (y - p3[1])² = r3²
    A = np.array([
        [2 * (pole_p2[0] - pole_p1[0]), 2 * (pole_p2[1] - pole_p1[1])],
        [2 * (pole_p3[0] - pole_p1[0]), 2 * (pole_p3[1] - pole_p1[1])]
    ])
    
    B = np.array([
        dist_p1_mm**2 - dist_p2_mm**2 + pole_p2[0]**2 - pole_p1[0]**2 + pole_p2[1]**2 - pole_p1[1]**2,
        dist_p1_mm**2 - dist_p3_mm**2 + pole_p3[0]**2 - pole_p1[0]**2 + pole_p3[1]**2 - pole_p1[1]**2
    ])
    
    # Check determinant: if poles are nearly collinear, det ≈ 0
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-6:
        return None
    
    try:
        robot_position = np.linalg.solve(A, B)
        return robot_position
    except np.linalg.LinAlgError:
        return None


def estimate_heading_from_pole(
    robot_position: np.ndarray,
    pole_position: np.ndarray,
    angle_lidar_deg: float,
) -> float:
    """
    Estimate robot heading from a single pole measurement.
    
    Given the measured angle to a pole in LIDAR frame,
    and the known pole position, compute the robot's absolute heading.
    
    Principle: angle_measured_lidar = angle_absolute - heading
              => heading = angle_absolute + angle_measured_lidar
    
    Args:
        robot_position: Robot [x, y] position (mm).
        pole_position: Pole [x, y] position (mm).
        angle_lidar_deg: Measured angle to pole in LIDAR frame (degrees).
    
    Returns:
        float: Estimated heading in radians (absolute, trigonometric).
    """
    # Absolute angle from robot to pole (trigonometric frame)
    angle_absolute = np.arctan2(
        pole_position[1] - robot_position[1],
        pole_position[0] - robot_position[0]
    )
    
    # Convert LIDAR angle to radians and compute heading
    angle_lidar_rad = np.radians(angle_lidar_deg)
    heading = angle_absolute + angle_lidar_rad
    
    return heading


def average_headings(
    heading_p1: float,
    heading_p2: float,
    heading_p3: float,
) -> float:
    """
    Compute circular mean of three heading estimates.
    
    Avoid direct averaging of angles (e.g., 359° + 1° = 180°),
    use sin/cos decomposition instead.
    
    Args:
        heading_p1: Heading estimate from pole 1 (rad).
        heading_p2: Heading estimate from pole 2 (rad).
        heading_p3: Heading estimate from pole 3 (rad).
    
    Returns:
        float: Circular mean heading in radians.
    """
    sin_sum = (np.sin(heading_p1) + np.sin(heading_p2) + np.sin(heading_p3)) / 3
    cos_sum = (np.cos(heading_p1) + np.cos(heading_p2) + np.cos(heading_p3)) / 3
    
    return np.arctan2(sin_sum, cos_sum)
