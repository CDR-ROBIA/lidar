"""
Dynamic mapping visualizer.

Keeps a time-decayed occupancy grid from LiDAR scans and highlights
moving objects by comparing recent observations to the background map.

Run with `--real --port /dev/ttyUSB0` to use the RPLidar device,
or without `--real` to simulate moving objects for demo.
"""
from __future__ import annotations
import time
import argparse
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    from map_matcher import EnvironmentMap
    from scan_processor import ScanProcessor
except Exception:
    # when executed as script, ensure project root is on path
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from map_matcher import EnvironmentMap
    from scan_processor import ScanProcessor

try:
    from lidar.lidar import Lidar, find_lidar_port
except Exception:
    Lidar = None
    find_lidar_port = None


class DynamicMapper:
    def __init__(self, env_map: EnvironmentMap, grid_cell_mm: float = 50.0, decay_s: float = 4.0):
        self.env = env_map
        self.cell = grid_cell_mm
        # time of last observation per cell (seconds since epoch), 0=never
        rows, cols = self.env.occupancy_grid.shape
        self.last_seen = np.zeros((rows, cols), dtype=float)
        self.decay_s = decay_s
        self.sp = ScanProcessor()
        self.prev_points_world = np.empty((0, 2), dtype=np.float32)

    def point_to_cell(self, x_mm: float, y_mm: float):
        c = int(x_mm / self.cell)
        r = int(y_mm / self.cell)
        return r, c

    def update_from_scan(self, scan_points_robot: np.ndarray, robot_pose: tuple):
        """Update last_seen grid using scan points in robot frame and robot_pose (x,y,theta_deg)."""
        if scan_points_robot is None or len(scan_points_robot) == 0:
            return np.empty((0, 2)), np.empty((0, 2))

        rx, ry, rtheta = robot_pose
        theta = math.radians(rtheta)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # transform to world coords
        world = np.zeros_like(scan_points_robot)
        world[:, 0] = scan_points_robot[:, 0] * cos_t - scan_points_robot[:, 1] * sin_t + rx
        world[:, 1] = scan_points_robot[:, 0] * sin_t + scan_points_robot[:, 1] * cos_t + ry

        now = time.time()
        rows, cols = self.last_seen.shape

        # mark last seen time per cell
        for x, y in world:
            r, c = self.point_to_cell(x, y)
            if 0 <= r < rows and 0 <= c < cols:
                self.last_seen[r, c] = now

        # detect moving points: those that appear in a cell that was empty recently (not within decay window)
        moving_mask = []
        static_mask = []
        for x, y in world:
            r, c = self.point_to_cell(x, y)
            if 0 <= r < rows and 0 <= c < cols:
                age = now - self.last_seen[r, c]
                # If we've seen this cell recently (within decay), consider static; else moving
                if age <= self.decay_s:
                    static_mask.append([x, y])
                else:
                    moving_mask.append([x, y])
            else:
                moving_mask.append([x, y])

        moving = np.array(moving_mask, dtype=np.float32) if moving_mask else np.empty((0, 2), dtype=np.float32)
        static = np.array(static_mask, dtype=np.float32) if static_mask else np.empty((0, 2), dtype=np.float32)

        self.prev_points_world = world
        return static, moving


def simulate_moving_objects(env: EnvironmentMap, robot_pose: tuple, t: float):
    """Simulate moving objects: returns scan points in robot frame for time t."""
    # Two moving objects: a person crossing and a box moving along x
    rx, ry, rtheta = robot_pose
    pts = []
    # person: moves along y near x=1200
    px = 1200
    py = 200 + (t * 150) % 1600
    # box: moves along x near y=800
    bx = 100 + (t * 100) % 2600
    by = 800

    for dx in np.linspace(-200, 200, 30):
        pts.append([px + dx - rx, py - ry])
    for dx in np.linspace(-150, 150, 25):
        pts.append([bx + dx - rx, by - ry])

    # add some static wall points from the environment (sampled)
    for w in env.walls:
        x1, y1, x2, y2 = w['x1'], w['y1'], w['x2'], w['y2']
        for t0 in np.linspace(0, 1, 8):
            wx = x1 + (x2 - x1) * t0
            wy = y1 + (y2 - y1) * t0
            pts.append([wx - rx, wy - ry])

    arr = np.array(pts, dtype=np.float32)
    # rotate by -rtheta to robot frame already done via subtraction
    theta = math.radians(-rtheta)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    out = np.zeros_like(arr)
    out[:, 0] = arr[:, 0] * cos_t - arr[:, 1] * sin_t
    out[:, 1] = arr[:, 0] * sin_t + arr[:, 1] * cos_t
    return out


def run_dynamic(env_map: EnvironmentMap, robot_pose: tuple, real=False, port=None):
    dm = DynamicMapper(env_map, grid_cell_mm=env_map.grid_cell_mm, decay_s=3.0)
    fig, ax = plt.subplots(figsize=(10, 6))
    rows, cols = env_map.occupancy_grid.shape

    im = ax.imshow(np.zeros((rows, cols)), cmap='viridis', origin='lower', extent=[0, env_map.width_mm, 0, env_map.height_mm])
    scan_scatter = ax.scatter([], [], s=8, color='white')
    moving_scatter = ax.scatter([], [], s=20, color='red')
    ax.set_xlim(0, env_map.width_mm)
    ax.set_ylim(0, env_map.height_mm)
    ax.set_title('LiDAR centered dynamic mapping')

    # robot marker at center
    robot_marker = ax.scatter([robot_pose[0]], [robot_pose[1]], color='red', s=100)


    if real and Lidar is not None:
        lidar = Lidar(port)
        lidar.start()
        gen = None
    else:
        lidar = None
        gen = None

    t0 = time.time()

    def update(frame_index):
        t = time.time() - t0
        if lidar is not None:
            raw = lidar.get_raw_scan()
            pts = []
            for q, ang, dist in raw:
                if dist > 0:
                    angr = math.radians(ang)
                    x = dist * math.cos(angr)
                    y = dist * math.sin(angr)
                    pts.append([x, y])
            scan = np.array(pts, dtype=np.float32) if pts else np.empty((0, 2), dtype=np.float32)
        else:
            scan = simulate_moving_objects(env_map, robot_pose, t)

        static_pts, moving_pts = dm.update_from_scan(scan, robot_pose)

        # occupancy visualization: recentness normalized
        now = time.time()
        ages = now - dm.last_seen
        vis = np.clip(1.0 - (ages / dm.decay_s), 0.0, 1.0)
        im.set_data(vis)

        if static_pts.size:
            # convert static world points back to world coords
            ax_pts = static_pts
            scan_scatter.set_offsets(ax_pts)
        else:
            scan_scatter.set_offsets(np.empty((0, 2)))

        if moving_pts.size:
            moving_scatter.set_offsets(moving_pts)
        else:
            moving_scatter.set_offsets(np.empty((0, 2)))

        ax.set_xlabel(f'Time: {t:.1f}s  moving:{len(moving_pts)} static:{len(static_pts)}')
        return im, scan_scatter, moving_scatter

    ani = FuncAnimation(fig, update, interval=200, blit=False)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--port', type=str, default=None)
    args = parser.parse_args()

    env = EnvironmentMap(width_mm=3000.0, height_mm=2000.0, grid_cell_mm=50.0)
    env.add_wall(100.0, 100.0, 2900.0, 100.0)
    env.add_wall(2900.0, 100.0, 2900.0, 1900.0)
    env.add_wall(2900.0, 1900.0, 100.0, 1900.0)
    env.add_wall(100.0, 1900.0, 100.0, 100.0)
    env.add_wall(1500.0, 100.0, 1500.0, 1000.0)
    env.add_landmark('dock', 200.0, 200.0)

    robot_pose = (1500.0, 1000.0, 0.0)
    run_dynamic(env, robot_pose, real=args.real, port=args.port)


if __name__ == '__main__':
    main()
