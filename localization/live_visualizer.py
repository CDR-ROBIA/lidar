"""
Live visualization for localization system.

Displays an animated window that progressively reveals LiDAR scan
points and updates wall detection and pose estimates. Uses a simulated
environment by default; can be adapted to read actual LiDAR scans.
"""
from __future__ import annotations
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from map_matcher import EnvironmentMap, MapMatcher
from scan_processor import ScanProcessor
from pose_estimator import PoseEstimator
import os

try:
    from lidar.lidar import Lidar, find_lidar_port
except Exception:
    Lidar = None
    find_lidar_port = None
    # Try adding project root to sys.path and import again
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from lidar.lidar import Lidar, find_lidar_port  # type: ignore
    except Exception:
        Lidar = None
        find_lidar_port = None


def generate_incremental_scan(env_map: EnvironmentMap, robot_pose: tuple, step_mm: float = 80.0):
    """Yield partial scan arrays to simulate streaming LiDAR."""
    # Generate full set of wall points in robot frame
    rx, ry, rtheta = robot_pose
    full_pts = []
    for w in env_map.walls:
        x1, y1, x2, y2 = w['x1'], w['y1'], w['x2'], w['y2']
        length = int(max(1, np.hypot(x2 - x1, y2 - y1) // step_mm))
        for i in range(length + 1):
            t = i / max(1, length)
            wx = x1 + (x2 - x1) * t
            wy = y1 + (y2 - y1) * t
            dx = wx - rx
            dy = wy - ry
            theta_rad = np.radians(-rtheta)
            lx = dx * np.cos(theta_rad) - dy * np.sin(theta_rad)
            ly = dx * np.sin(theta_rad) + dy * np.cos(theta_rad)
            full_pts.append([lx, ly])

    full = np.array(full_pts, dtype=np.float32)
    # Shuffle to simulate unordered scan
    idx = np.arange(len(full))
    np.random.shuffle(idx)
    full = full[idx]

    # Yield increasing portions
    for end in range(5, len(full) + 5, 5):
        yield full[:min(end, len(full))]


def generate_from_lidar(lidar: Lidar):
    """Yield scans read from a running `Lidar` instance as arrays in robot frame."""
    if lidar is None:
        return

    try:
        while True:
            raw = lidar.get_raw_scan()
            if raw:
                # Convert to (x,y) robot frame
                pts = []
                for quality, angle_deg, dist in raw:
                    if dist <= 0:
                        continue
                    angle_rad = np.radians(angle_deg)
                    x = dist * np.cos(angle_rad)
                    y = dist * np.sin(angle_rad)
                    pts.append([x, y])
                yield np.array(pts, dtype=np.float32)
            else:
                # No data yet
                yield np.empty((0, 2), dtype=np.float32)

            time.sleep(0.05)
    except GeneratorExit:
        return


def run_live(env_map: EnvironmentMap, robot_pose: tuple, frames=None):
    sp = ScanProcessor()
    matcher = MapMatcher(env_map)
    estimator = PoseEstimator(*robot_pose)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, env_map.width_mm)
    ax.set_ylim(0, env_map.height_mm)
    ax.set_aspect('equal', 'box')
    ax.set_title('Live Localization')

    # Draw static map elements
    grid = env_map.occupancy_grid.copy()
    ax.imshow(grid[::-1, :], cmap='Greys', extent=[0, env_map.width_mm, 0, env_map.height_mm])
    for w in env_map.walls:
        ax.plot([w['x1'], w['x2']], [w['y1'], w['y2']], color='orange', linewidth=2)
    for name, (lx, ly) in env_map.landmarks.items():
        ax.scatter([lx], [ly], marker='*', color='cyan', s=100)
        ax.text(lx + 20, ly + 20, name, color='cyan')

    robot_scatter = ax.scatter([], [], color='red', s=80)
    heading_line, = ax.plot([], [], color='red')
    scan_scatter = ax.scatter([], [], s=6, color='lime')
    text_info = ax.text(10, env_map.height_mm - 20, '', color='black')

    gen = frames if frames is not None else generate_incremental_scan(env_map, robot_pose, step_mm=80.0)

    def init():
        robot_scatter.set_offsets(np.empty((0, 2)))
        scan_scatter.set_offsets(np.empty((0, 2)))
        heading_line.set_data([], [])
        text_info.set_text('')
        return robot_scatter, scan_scatter, heading_line, text_info

    def update(frame):
        # frame is array of points in robot frame
        processed = sp.process_scan([(10, float(np.degrees(np.arctan2(y, x))), float(np.hypot(x, y))) for x, y in frame])
        # Update pose estimator (no movement, but we can run a fake match)
        match = matcher.match_scan(processed, estimator.get_position_mm() + (estimator.get_orientation_deg(),), search_radius_mm=100.0, angle_step_deg=10.0)
        best = match.get('best_match', estimator.get_position_mm() + (estimator.get_orientation_deg(),))
        # Apply scan match correction (difference)
        bx, by, btheta = best
        ex, ey = estimator.get_position_mm()
        etheta = estimator.get_orientation_deg()
        dx = bx - ex
        dy = by - ey
        dtheta = btheta - etheta
        estimator.update_from_scan_match(dx, dy, dtheta, confidence=match.get('confidence', 0.5))

        # Update robot marker and heading
        rx, ry = estimator.get_position_mm()
        rtheta = estimator.get_orientation_deg()
        robot_scatter.set_offsets([[rx, ry]])
        theta_rad = np.radians(rtheta)
        hx = rx + 200.0 * np.cos(theta_rad)
        hy = ry + 200.0 * np.sin(theta_rad)
        heading_line.set_data([rx, hx], [ry, hy])

        # Convert processed scan to world for plotting
        if processed.size:
            cos_t = np.cos(np.radians(rtheta))
            sin_t = np.sin(np.radians(rtheta))
            world = np.zeros_like(processed)
            world[:, 0] = processed[:, 0] * cos_t - processed[:, 1] * sin_t + rx
            world[:, 1] = processed[:, 0] * sin_t + processed[:, 1] * cos_t + ry
            scan_scatter.set_offsets(world)
        else:
            scan_scatter.set_offsets([])

        info = f"Pose: ({int(rx)},{int(ry)}) θ={int(rtheta)}°  Confidence={estimator.get_pose().confidence:.2f}"
        text_info.set_text(info)

        return robot_scatter, scan_scatter, heading_line, text_info

    ani = FuncAnimation(fig, update, frames=gen, init_func=init, interval=300, blit=False, repeat=False)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Live visualizer for localization')
    parser.add_argument('--real', action='store_true', help='Use real LiDAR hardware')
    parser.add_argument('--port', type=str, default=None, help='Serial port for LiDAR')
    args = parser.parse_args()

    env = EnvironmentMap(width_mm=3000.0, height_mm=2000.0, grid_cell_mm=50.0)
    env.add_wall(100.0, 100.0, 2900.0, 100.0)
    env.add_wall(2900.0, 100.0, 2900.0, 1900.0)
    env.add_wall(2900.0, 1900.0, 100.0, 1900.0)
    env.add_wall(100.0, 1900.0, 100.0, 100.0)
    env.add_wall(1500.0, 100.0, 1500.0, 1000.0)
    env.add_landmark("dock", 200.0, 200.0)
    env.add_landmark("charging", 2800.0, 1800.0)

    if args.real:
        if Lidar is None:
            print('LiDAR support not available in this environment.')
            return

        port = args.port or (find_lidar_port() if find_lidar_port else None)
        if port is None:
            print('No LiDAR port found. Use --port to specify one.')
            return

        lidar_dev = Lidar(port)
        try:
            lidar_dev.start()
            print(f'LiDAR started on {port}')
            # Use real scans and positions from the device
            gen = generate_from_lidar(lidar_dev)

            # Use initial pose from device if available
            pos = lidar_dev.get_pos() or (500.0, 600.0, 0.0)
            # pos heading is radians in lidar.Lidar
            robot_pose = (pos[0], pos[1], np.degrees(pos[2]))
            run_live(env, robot_pose, frames=gen)
        finally:
            lidar_dev.stop()
    else:
        robot_pose = (500.0, 600.0, 30.0)
        run_live(env, robot_pose)


if __name__ == '__main__':
    main()
