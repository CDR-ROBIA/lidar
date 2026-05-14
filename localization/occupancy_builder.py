"""
Occupancy grid builder

Builds a 2D occupancy grid from LiDAR scans using a log-odds update rule.
The LiDAR is fixed at the center cell (0,0 origin) and the map grows around it.
Supports real LiDAR (`--real`) or a simple simulated moving environment.

Visualization approximates the style of the attached image: dark background,
occupied areas filled (cyan), free space dark, robot shown as red marker.
"""
from __future__ import annotations
import argparse
import math
import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    from lidar.lidar import Lidar, find_lidar_port
except Exception:
    Lidar = None
    find_lidar_port = None


class OccupancyBuilder:
    def __init__(self, width_mm=6000, height_mm=4000, cell_mm=50):
        self.cell_mm = cell_mm
        self.cols = int(width_mm // cell_mm)
        self.rows = int(height_mm // cell_mm)
        # log-odds grid, 0 = unknown (p=0.5)
        self.logodds = np.zeros((self.rows, self.cols), dtype=float)
        # counters for times marked occupied
        self.occ_count = np.zeros((self.rows, self.cols), dtype=int)
        # center index (LiDAR position)
        self.center_r = self.rows // 2
        self.center_c = self.cols // 2
        # parameters
        self.lo_occ = 1.2  # increment for occupied cell
        self.lo_free = -0.2  # decrement for free cell along beam
        self.lo_max = 20.0
        self.lo_min = -20.0
        # cleaning parameters
        self.decay_factor = 0.995  # temporal smoothing of log-odds
        self.min_cluster_size = 8  # remove clusters smaller than this (cells)

    def world_to_cell(self, x_mm: float, y_mm: float):
        # x_mm positive to the right, y_mm positive up
        c = int(round(x_mm / self.cell_mm)) + self.center_c
        r = self.center_r - int(round(y_mm / self.cell_mm))
        return r, c

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def bresenham(self, r0, c0, r1, c1):
        # integer Bresenham line from (r0,c0) to (r1,c1)
        points = []
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dc - dr if dr > dc else dr - dc
        r, c = r0, c0
        if dr > dc:
            err = dr // 2
            while r != r1:
                points.append((r, c))
                r += sr
                err -= dc
                if err < 0:
                    c += sc
                    err += dr
            points.append((r1, c1))
        else:
            err = dc // 2
            while c != c1:
                points.append((r, c))
                c += sc
                err -= dr
                if err < 0:
                    r += sr
                    err += dc
            points.append((r1, c1))
        return points

    def update_scan(self, scan_points_robot: np.ndarray):
        # scan_points_robot: Nx2 array in robot frame (mm)
        if scan_points_robot is None or len(scan_points_robot) == 0:
            return

        r0, c0 = self.center_r, self.center_c

        for x, y in scan_points_robot:
            # convert to world coords relative to lidar at center
            # here robot frame == world frame since lidar fixed at center
            r, c = self.world_to_cell(x, y)
            if not self.in_bounds(r, c):
                continue

            # trace beam from center to endpoint
            line = self.bresenham(r0, c0, r, c)
            # mark free cells along beam (exclude endpoint)
            for (rr, cc) in line[:-1]:
                self.logodds[rr, cc] += self.lo_free
                if self.logodds[rr, cc] < self.lo_min:
                    self.logodds[rr, cc] = self.lo_min

            # mark endpoint occupied and densify nearby cells to increase visible points
            # densify in a 5x5 neighborhood around endpoint for more visible coverage
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    rr = r + dr
                    cc = c + dc
                    if not self.in_bounds(rr, cc):
                        continue
                    # smaller increment for neighbors further from center
                    dist = max(abs(dr), abs(dc))
                    scale = 1.0 - 0.2 * dist
                    if scale <= 0:
                        continue
                    inc = self.lo_occ * scale
                    self.logodds[rr, cc] += inc
                    if self.logodds[rr, cc] > self.lo_max:
                        self.logodds[rr, cc] = self.lo_max
                    if dr == 0 and dc == 0:
                        self.occ_count[rr, cc] += 1

    def get_probability(self):
        # convert log-odds to probability p = 1/(1+exp(-l))
        p = 1.0 / (1.0 + np.exp(-self.logodds))
        return p

    def get_display(self):
        # produce RGB image like the attachment: occupied -> cyan, unknown/dark for free
        p = self.get_probability()
        img = np.zeros((self.rows, self.cols, 3), dtype=float)
        # background dark blue-ish
        img[:, :, 0] = 0.05
        img[:, :, 1] = 0.08
        img[:, :, 2] = 0.12

        # occupied cells (p > 0.6) colored cyan
        occ_mask = p > 0.6
        img[occ_mask, 0] = 0.3
        img[occ_mask, 1] = 0.92
        img[occ_mask, 2] = 0.86

        # free cells (p < 0.4) slightly darker
        free_mask = p < 0.4
        img[free_mask, 0] = 0.02
        img[free_mask, 1] = 0.04
        img[free_mask, 2] = 0.06

        return img

    def _connected_components(self, mask: np.ndarray):
        """Simple 4-connected components labeling. Returns list of component cell lists."""
        rows, cols = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        comps = []
        for r in range(rows):
            for c in range(cols):
                if not mask[r, c] or visited[r, c]:
                    continue
                # BFS
                stack = [(r, c)]
                comp = []
                visited[r, c] = True
                while stack:
                    rr, cc = stack.pop()
                    comp.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and mask[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                comps.append(comp)
        return comps

    def remove_small_clusters(self):
        """Remove small occupied clusters by lowering their log-odds (make them free)."""
        p = self.get_probability()
        occ_mask = p > 0.6
        comps = self._connected_components(occ_mask)
        for comp in comps:
            if len(comp) < self.min_cluster_size:
                for (r, c) in comp:
                    # decrease logodds to push toward free
                    self.logodds[r, c] += self.lo_free * 2.5
                    if self.logodds[r, c] < self.lo_min:
                        self.logodds[r, c] = self.lo_min


def simulate_environment(t, max_range=4000):
    # simulate some random obstacles moving around the sensor
    pts = []
    # a denser ring of static points further out (1deg resolution)
    for ang in range(0, 360, 1):
        r = 2000 + 200 * math.sin(math.radians(3 * t + ang))
        a = math.radians(ang)
        x = r * math.cos(a)
        y = r * math.sin(a)
        pts.append([x, y])

    # moving object: denser point cloud
    mx = 500 * math.cos(t / 2.0)
    my = 500 * math.sin(t / 2.0)
    for d in np.linspace(-80, 80, 80):
        pts.append([mx + d, my + (d * 0.25)])

    return np.array(pts, dtype=float)


def run_builder(real=False, port=None):
    # use finer resolution for better coherence
    builder = OccupancyBuilder(width_mm=6000, height_mm=4000, cell_mm=25)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('#0b0f14')
    im = ax.imshow(builder.get_display(), origin='lower', extent=[-builder.cols // 2 * builder.cell_mm,
                                                                   builder.cols // 2 * builder.cell_mm,
                                                                   -builder.rows // 2 * builder.cell_mm,
                                                                   builder.rows // 2 * builder.cell_mm])
    robot_dot, = ax.plot(0, 0, 'o', color='red', markersize=8)
    ax.set_title('Occupancy Builder (LiDAR at center)')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')

    lidar = None
    if real and Lidar is not None:
        lidar = Lidar(port)
        lidar.start()

    t0 = time.time()

    def update(frame):
        t = time.time() - t0
        if lidar is not None:
            raw = lidar.get_raw_scan()
            pts = []
            for q, ang, dist in raw:
                if dist <= 0 or dist > 6000:
                    continue
                a = math.radians(ang)
                x = dist * math.cos(a)
                y = dist * math.sin(a)
                pts.append([x, y])
            scan = np.array(pts, dtype=float) if pts else np.empty((0, 2), dtype=float)
        else:
            scan = simulate_environment(t)

        builder.update_scan(scan)
        img = builder.get_display()
        im.set_data(img)
        ax.set_title(f'Occupancy Builder (t={t:.1f}s)')
        return im,

    ani = FuncAnimation(fig, update, interval=300, blit=False)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--port', type=str, default=None)
    args = parser.parse_args()

    port = args.port
    if args.real and port is None and find_lidar_port is not None:
        port = find_lidar_port()

    run_builder(real=args.real, port=port)


if __name__ == '__main__':
    main()
