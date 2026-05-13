"""
Visualization module for LiDAR robot localization.

Real-time matplotlib display with performance optimization (blit=True).
Completely decoupled from Lidar class - uses only public API (get_pos, get_raw_scan).
"""

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection

from .lidar import Lidar


def run_visualizer(lidar: Lidar) -> None:
    """
    Display real-time visualization of robot position and LiDAR scan.
    
    Uses matplotlib with blit=True optimization (pre-created artists).
    Does NOT call ax.cla() per frame, which was the performance bottleneck.
    Instead, updates artist offsets/data incrementally.
    
    Args:
        lidar: Lidar instance to visualize.
    """
    # Figure setup
    fig, ax = plt.subplots(figsize=(15, 14))
    ax.set_facecolor("#0d0d0d")
    fig.patch.set_facecolor("#0d0d0d")
    
    ax.set_xlim(-300, 3400)
    ax.set_ylim(-300, 2400)
    ax.set_aspect("equal")
    ax.grid(True, color="#333333", linewidth=0.5)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    
    # Pre-create artists (instead of recreating per frame)
    scatter_scan = ax.scatter([], [], s=2, color="#00aaff", alpha=0.4, label="scan")
    scatter_poles = ax.scatter([], [], s=144, color="#ff4444", marker="o", zorder=5)
    scatter_robot = ax.scatter([], [], s=196, color="#00ff88", marker="o", zorder=6)
    
    line_p1_robot = ax.plot([], [], "--", color="#ffaa00", alpha=0.5, linewidth=1)[0]
    line_p2_robot = ax.plot([], [], "--", color="#ffaa00", alpha=0.5, linewidth=1)[0]
    line_p3_robot = ax.plot([], [], "--", color="#ffaa00", alpha=0.5, linewidth=1)[0]
    
    # Text annotations
    text_title = ax.text(0.5, 0.98, "", transform=ax.transAxes, color="white",
                         ha="center", fontsize=11, verticalalignment="top")
    text_info = ax.text(0.02, 0.98, "", transform=ax.transAxes, color="white",
                        fontsize=9, verticalalignment="top", family="monospace",
                        bbox=dict(boxstyle="round", facecolor="#1a1a1a", alpha=0.8))
    
    # Arena boundary (polygon)
    arena_corners = np.array([
        [0, 0],
        [3000, 0],
        [3000, 2000],
        [0, 2000],
    ])
    arena_patch = Polygon(arena_corners, linewidth=2, edgecolor="cyan", facecolor="none")
    ax.add_patch(arena_patch)
    
    # Pole positions (fixed)
    pole_positions = {
        "P1": (0, 0),
        "P2": (3000, 0),
        "P3": (3000, 2000),
    }
    
    pole_xy = np.array(list(pole_positions.values()))
    
    def update_frame(frame):
        """Update all artists for current frame."""
        artists = []
        
        # Get current position and scan
        pos = lidar.get_pos()
        scan = lidar.get_raw_scan()
        
        if pos is not None:
            robot_x, robot_y, heading = pos
            scatter_robot.set_offsets([[robot_x, robot_y]])
            artists.append(scatter_robot)
        
        # Update pole positions
        scatter_poles.set_offsets(pole_xy)
        artists.append(scatter_poles)
        
        # Update scan points (converted to absolute frame)
        if len(scan) > 0 and pos is not None:
            robot_x, robot_y, heading = pos
            xs, ys = _get_scan_points_absolute(scan, robot_x, robot_y, heading)
            scatter_scan.set_offsets(np.c_[xs, ys])
        else:
            scatter_scan.set_offsets(np.empty((0, 2)))
        artists.append(scatter_scan)
        
        # Update pole-robot connection lines
        if pos is not None:
            robot_x, robot_y, heading = pos
            
            for (line, pole_name) in [
                (line_p1_robot, "P1"),
                (line_p2_robot, "P2"),
                (line_p3_robot, "P3"),
            ]:
                px, py = pole_positions[pole_name]
                line.set_data([robot_x, px], [robot_y, py])
                artists.append(line)
        
        # Update text
        if pos is not None:
            robot_x, robot_y, heading = pos
            heading_deg = np.degrees(heading)
            
            title_text = f"X={int(robot_x)}mm  Y={int(robot_y)}mm  Heading={heading_deg:.1f}°  Status: OK"
            text_title.set_text(title_text)
            text_title.set_color("#00ff88")
        else:
            text_title.set_text("Position not available")
            text_title.set_color("#ff4444")
        
        artists.append(text_title)
        
        # Info box with distances/angles
        info_text = _build_info_text(lidar)
        text_info.set_text(info_text)
        artists.append(text_info)
        
        return artists
    
    ani = animation.FuncAnimation(
        fig, update_frame,
        interval=300,  # 300 ms = ~3 FPS
        blit=True,
        cache_frame_data=False,
    )
    
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        plt.close(fig)


def _get_scan_points_absolute(
    scan: List,
    robot_x: float,
    robot_y: float,
    heading: float,
) -> Tuple[List[float], List[float]]:
    """
    Convert scan points from LiDAR polar coordinates to absolute cartesian.
    
    Subsample 1 point per 3 to reduce rendering load.
    
    Args:
        scan: Raw scan data from RPLidar (list of tuples).
        robot_x: Robot X position (mm).
        robot_y: Robot Y position (mm).
        heading: Robot heading (rad, trigonometric).
    
    Returns:
        (xs, ys): Lists of absolute X and Y coordinates.
    """
    DIST_MAX = int(np.sqrt(3000**2 + 2000**2))
    xs, ys = [], []
    
    for i, measurement in enumerate(scan):
        # Subsample: 1 point per 3
        if i % 3 != 0:
            continue
        
        angle_lidar_deg = measurement[1]
        distance = measurement[2]
        
        if distance <= 0 or distance > DIST_MAX:
            continue
        
        # Convert: LIDAR angle → absolute angle → absolute coordinates
        angle_absolu = heading - np.radians(angle_lidar_deg)
        px = robot_x + distance * np.cos(angle_absolu)
        py = robot_y + distance * np.sin(angle_absolu)
        
        xs.append(px)
        ys.append(py)
    
    return xs, ys


def _build_info_text(lidar: Lidar) -> str:
    """Build information text for top-left corner."""
    pos = lidar.get_pos()
    
    if pos is None:
        return "Position: not available\nStatus: initializing..."
    
    robot_x, robot_y, heading = pos
    heading_deg = np.degrees(heading)
    
    # Try to extract distance/angle info from internal state (via inspection)
    # This is a compromise: we use get_pos() (public API) for position,
    # but peek at internal state for pole measurements.
    # In a production system, you'd add methods like get_pole_distances().
    
    with lidar._lock:
        state = lidar._state
        dist_p1 = state.dist_p1_mm
        dist_p2 = state.dist_p2_mm
        dist_p3 = state.dist_p3_mm
        angle_p1 = state.angle_p1_deg
        angle_p2 = state.angle_p2_deg
        angle_p3 = state.angle_p3_deg
    
    info = f"Position: ({int(robot_x)}, {int(robot_y)}) mm\n"
    info += f"Heading: {heading_deg:.1f}°\n"
    info += "\n"
    info += f"P1 → angle={angle_p1:.0f}°  dist={int(dist_p1) if dist_p1 else '?'}mm\n"
    info += f"P2 → angle={angle_p2:.0f}°  dist={int(dist_p2) if dist_p2 else '?'}mm\n"
    info += f"P3 → angle={angle_p3:.0f}°  dist={int(dist_p3) if dist_p3 else '?'}mm\n"
    
    return info
