"""
Simple visualization tool for the localization system.

Generates a demo environment, simulates a LiDAR scan from the robot
pose, and plots the occupancy grid, known walls/landmarks, the robot
pose, and LiDAR points. Saves a snapshot `visualization.png`.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from map_matcher import EnvironmentMap
from scan_processor import ScanProcessor
from pose_estimator import PoseEstimator
import os


def create_demo_map() -> EnvironmentMap:
    m = EnvironmentMap(width_mm=3000.0, height_mm=2000.0, grid_cell_mm=50.0)
    # Simple rectangular room
    m.add_wall(100.0, 100.0, 2900.0, 100.0)
    m.add_wall(2900.0, 100.0, 2900.0, 1900.0)
    m.add_wall(2900.0, 1900.0, 100.0, 1900.0)
    m.add_wall(100.0, 1900.0, 100.0, 100.0)
    # Add an internal wall
    m.add_wall(1500.0, 100.0, 1500.0, 1000.0)
    m.add_landmark("dock", 200.0, 200.0)
    m.add_landmark("charging", 2800.0, 1800.0)
    return m


def simulate_scan_from_walls(env_map: EnvironmentMap, robot_pose: tuple, step_mm: float = 100.0) -> np.ndarray:
    """Generate scan points sampled along walls, expressed in robot-local frame.

    Args:
        env_map: EnvironmentMap with walls
        robot_pose: (x_mm, y_mm, theta_deg)
        step_mm: spacing between sampled points on walls
    Returns:
        numpy array (N,2) of points in robot frame
    """
    rx, ry, rtheta = robot_pose
    pts = []
    for w in env_map.walls:
        x1, y1, x2, y2 = w['x1'], w['y1'], w['x2'], w['y2']
        length = int(max(1, np.hypot(x2 - x1, y2 - y1) // step_mm))
        for i in range(length + 1):
            t = i / max(1, length)
            wx = x1 + (x2 - x1) * t
            wy = y1 + (y2 - y1) * t
            # Convert to robot-local coordinates
            dx = wx - rx
            dy = wy - ry
            theta_rad = np.radians(-rtheta)
            lx = dx * np.cos(theta_rad) - dy * np.sin(theta_rad)
            ly = dx * np.sin(theta_rad) + dy * np.cos(theta_rad)
            pts.append([lx, ly])

    return np.array(pts, dtype=np.float32)


def plot_snapshot(env_map: EnvironmentMap, pose: tuple, scan_points: np.ndarray, out_file: str = "visualization.png") -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    # Occupancy grid (display origin at lower-left)
    grid = env_map.occupancy_grid.copy()
    ax.imshow(grid[::-1, :], cmap='Greys', extent=[0, env_map.width_mm, 0, env_map.height_mm])

    # Plot walls (world coords)
    for w in env_map.walls:
        ax.plot([w['x1'], w['x2']], [w['y1'], w['y2']], color='orange', linewidth=2)

    # Plot landmarks
    for name, (lx, ly) in env_map.landmarks.items():
        ax.scatter([lx], [ly], marker='*', color='cyan', s=120)
        ax.text(lx + 20, ly + 20, name, color='cyan')

    # Plot robot pose
    rx, ry, rtheta = pose
    ax.scatter([rx], [ry], color='red', s=80)
    # Draw heading
    heading_len = 200.0
    theta_rad = np.radians(rtheta)
    hx = rx + heading_len * np.cos(theta_rad)
    hy = ry + heading_len * np.sin(theta_rad)
    ax.plot([rx, hx], [ry, hy], color='red')

    # Transform scan points (robot frame) to world frame for plotting
    if scan_points is not None and len(scan_points) > 0:
        theta_rad = np.radians(rtheta)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        world = np.zeros_like(scan_points)
        world[:, 0] = scan_points[:, 0] * cos_t - scan_points[:, 1] * sin_t + rx
        world[:, 1] = scan_points[:, 0] * sin_t + scan_points[:, 1] * cos_t + ry
        ax.scatter(world[:, 0], world[:, 1], s=6, color='lime', alpha=0.8)

    ax.set_xlim(0, env_map.width_mm)
    ax.set_ylim(0, env_map.height_mm)
    ax.set_aspect('equal', 'box')
    ax.set_title('Localization Visualization')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')

    plt.tight_layout()
    plt.savefig(out_file)

    # Show interactively if possible
    try:
        plt.show()
    except Exception:
        pass


def main():
    env = create_demo_map()
    # Robot near left side
    robot_pose = (500.0, 600.0, 30.0)  # x_mm, y_mm, theta_deg

    # Simulate scan
    scan_pts = simulate_scan_from_walls(env, robot_pose, step_mm=80.0)

    # Filter scan using ScanProcessor (optional)
    sp = ScanProcessor()
    # Convert simulated (x,y) robot frame points to pseudo raw scan format
    # (quality, angle_deg, distance_mm)
    raw = []
    for x, y in scan_pts:
        angle = np.degrees(np.arctan2(y, x))
        dist = np.hypot(x, y)
        raw.append((10, float(angle), float(dist)))

    processed = sp.process_scan(raw)

    out_path = os.path.join(os.getcwd(), 'visualization.png')
    plot_snapshot(env, robot_pose, processed, out_file=out_path)
    print(f"Saved visualization snapshot to: {out_path}")


if __name__ == '__main__':
    main()
