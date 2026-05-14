"""
Symmetry and periodicity analyzer for occupancy grids.

Builds an occupancy grid (using OccupancyBuilder) from simulated or real
LiDAR scans, then attempts to detect axes of symmetry and periodic structures
in the environment. Outputs a diagnostic plot and prints scores.
"""
from __future__ import annotations
import time
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from occupancy_builder import OccupancyBuilder, simulate_environment
import os
import sys

try:
    from lidar.lidar import Lidar, find_lidar_port
except Exception:
    Lidar = None
    find_lidar_port = None
    # try adding project root to path and import again
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from lidar.lidar import Lidar, find_lidar_port  # type: ignore
    except Exception:
        Lidar = None
        find_lidar_port = None


def cell_to_world(builder: OccupancyBuilder, r: int, c: int):
    x = (c - builder.center_c) * builder.cell_mm
    y = (builder.center_r - r) * builder.cell_mm
    return x, y


def analyze_symmetry(builder: OccupancyBuilder):
    p = builder.get_probability()
    occ = p > 0.65
    rows, cols = occ.shape
    # get occupied cell world coordinates
    coords = []
    for r in range(rows):
        for c in range(cols):
            if occ[r, c]:
                coords.append(cell_to_world(builder, r, c))
    if not coords:
        print('No occupied cells found')
        return None
    pts = np.array(coords)
    centroid = pts.mean(axis=0)

    # symmetry test: reflect across line through centroid at angle theta
    def reflect_points(theta_rad):
        # unit vector along line
        ux = math.cos(theta_rad)
        uy = math.sin(theta_rad)
        a = centroid
        # reflect each point
        reflected = []
        for x, y in pts:
            vx = x - a[0]
            vy = y - a[1]
            proj = ux * (vx * ux + vy * uy) + uy * 0  # scalar projection along u times u? compute properly
            # Actually projection vector:
            dot = vx * ux + vy * uy
            projx = dot * ux
            projy = dot * uy
            rx = 2 * (a[0] + projx) - x
            ry = 2 * (a[1] + projy) - y
            reflected.append((rx, ry))
        return np.array(reflected)

    # build occupancy mask for fast lookup
    occ_mask = occ

    def world_to_cell_idx(x, y):
        c = int(round(x / builder.cell_mm)) + builder.center_c
        r = builder.center_r - int(round(y / builder.cell_mm))
        return r, c

    best_angle = None
    best_score = -1.0
    scores = []
    for deg in range(0, 180, 1):
        theta = math.radians(deg)
        ref = reflect_points(theta)
        matches = 0
        total = len(ref)
        for rx, ry in ref:
            rr, cc = world_to_cell_idx(rx, ry)
            if 0 <= rr < rows and 0 <= cc < cols and occ_mask[rr, cc]:
                matches += 1
        score = matches / total if total > 0 else 0
        scores.append(score)
        if score > best_score:
            best_score = score
            best_angle = deg

    print(f'Best symmetry angle (deg): {best_angle} score={best_score:.3f}')

    # periodicity via autocorrelation (FFT)
    occ_float = occ.astype(float)
    # subtract mean to center
    occ_centered = occ_float - occ_float.mean()
    # compute autocorrelation via FFT
    f = np.fft.fft2(occ_centered)
    ac = np.fft.ifft2(np.abs(f)**2).real
    ac = np.fft.fftshift(ac)
    # find peaks excluding center
    center = (ac.shape[0] // 2, ac.shape[1] // 2)
    ac[center[0]-2:center[0]+3, center[1]-2:center[1]+3] = 0
    peak_idx = np.unravel_index(np.argmax(ac), ac.shape)
    dy = peak_idx[0] - center[0]
    dx = peak_idx[1] - center[1]
    period_x = abs(dx) * builder.cell_mm
    period_y = abs(dy) * builder.cell_mm
    print(f'Detected dominant translation: dx_cells={dx}, dy_cells={dy} -> period_x={period_x}mm, period_y={period_y}mm')

    result = {
        'centroid': centroid,
        'best_angle_deg': best_angle,
        'best_score': best_score,
        'period_x_mm': period_x,
        'period_y_mm': period_y,
        'occ_mask': occ
    }
    return result


def run_simulation_and_analyze(duration_s=6.0, cell_mm=10):
    builder = OccupancyBuilder(width_mm=4000, height_mm=4000, cell_mm=cell_mm)
    t0 = time.time()
    while time.time() - t0 < duration_s:
        t = time.time() - t0
        pts = simulate_environment(t)
        builder.update_scan(pts)
        # apply light decay smoothing
        builder.logodds *= builder.decay_factor
        builder.remove_small_clusters()

    res = analyze_symmetry(builder)
    _plot_and_show(builder, res)


def run_real_and_analyze(duration_s=6.0, cell_mm=10, port=None):
    builder = OccupancyBuilder(width_mm=4000, height_mm=4000, cell_mm=cell_mm)
    lidar = None
    if Lidar is None:
        print('LiDAR library not available; cannot run in --real mode')
        return

    if port is None and find_lidar_port is not None:
        port = find_lidar_port()

    lidar = Lidar(port)
    try:
        lidar.start()
        print(f'Collecting scans from LiDAR on {port} for {duration_s}s...')
        t0 = time.time()
        while time.time() - t0 < duration_s:
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
            if len(scan):
                builder.update_scan(scan)
            # smoothing + cleanup per iteration
            builder.logodds *= builder.decay_factor
            builder.remove_small_clusters()
            time.sleep(0.05)

    finally:
        try:
            lidar.stop()
        except Exception:
            pass

    res = analyze_symmetry(builder)
    _plot_and_show(builder, res)


def _plot_and_show(builder, res, out_png=None, out_json=None):
    # plot
    fig, ax = plt.subplots(figsize=(8, 8))
    img = builder.get_display()
    ax.imshow(img, origin='lower', extent=[-builder.cols//2*builder.cell_mm, builder.cols//2*builder.cell_mm, -builder.rows//2*builder.cell_mm, builder.rows//2*builder.cell_mm])
    ax.plot(0, 0, 'ro')
    if res is not None:
        cx, cy = res['centroid']
        ang = math.radians(res['best_angle_deg'])
        # draw symmetry axis through centroid
        L = max(builder.cols, builder.rows) * builder.cell_mm
        dx = math.cos(ang) * L
        dy = math.sin(ang) * L
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color='yellow', linewidth=2)
        ax.set_title(f"Symmetry {res['best_angle_deg']}deg score={res['best_score']:.2f} period_x={res['period_x_mm']:.0f}mm")

    if out_png:
        fig.savefig(out_png)
        print(f'Saved plot to {out_png}')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=6.0)
    parser.add_argument('--cell', type=int, default=10)
    parser.add_argument('--real', action='store_true', help='Use real LiDAR')
    parser.add_argument('--port', type=str, default=None, help='Serial port for LiDAR')
    parser.add_argument('--out', type=str, default=None, help='Output basename for plot/json')
    args = parser.parse_args()
    if args.real:
        run_real_and_analyze(duration_s=args.duration, cell_mm=args.cell, port=args.port)
    else:
        run_simulation_and_analyze(duration_s=args.duration, cell_mm=args.cell)


if __name__ == '__main__':
    main()
