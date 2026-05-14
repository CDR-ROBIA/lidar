"""
Map builder runner

Collects LiDAR scans for a given duration, builds an occupancy grid using
`OccupancyBuilder`, applies stabilization (min observation threshold) and
simple morphological closing to remove noise, then saves the final map and
displays it.

Usage:
  .venv/bin/python localization/map_builder_runner.py --duration 30 --cell 25 --port /dev/ttyUSB0 --out map_run1
"""
from __future__ import annotations
import argparse
import time
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

try:
    from lidar.lidar import Lidar, find_lidar_port
except Exception:
    Lidar = None
    find_lidar_port = None
    # try adding project root and re-import
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from lidar.lidar import Lidar, find_lidar_port  # type: ignore
    except Exception:
        Lidar = None
        find_lidar_port = None

from occupancy_builder import OccupancyBuilder


def morphological_close(mask: np.ndarray, iters: int = 1):
    """Simple morphological closing: dilation followed by erosion using 4-neighborhood."""
    def dilate(m):
        rows, cols = m.shape
        out = m.copy()
        for r in range(rows):
            for c in range(cols):
                if m[r, c]:
                    out[max(0, r-1):min(rows, r+2), max(0, c-1):min(cols, c+2)] = True
        return out

    def erode(m):
        rows, cols = m.shape
        out = np.zeros_like(m)
        for r in range(rows):
            for c in range(cols):
                sub = m[max(0, r-1):min(rows, r+2), max(0, c-1):min(cols, c+2)]
                # keep if all neighbors (3x3) mostly true
                if sub.mean() > 0.5:
                    out[r, c] = True
        return out

    m = mask.copy()
    for _ in range(iters):
        m = dilate(m)
    for _ in range(iters):
        m = erode(m)
    return m


def run(duration_s: int = 30, cell_mm: int = 25, port: str | None = None, outname: str | None = None):
    builder = OccupancyBuilder(width_mm=6000, height_mm=4000, cell_mm=cell_mm)

    if port is None and find_lidar_port is not None:
        port = find_lidar_port()

    if Lidar is None:
        print('LiDAR module not available; cannot collect real scans.')
        return

    lidar = Lidar(port)
    # Prepare live plot
    fig, ax = plt.subplots(figsize=(10, 8))
    img_disp = builder.get_display()
    im = ax.imshow(img_disp, origin='lower', extent=[-builder.cols//2*builder.cell_mm, builder.cols//2*builder.cell_mm, -builder.rows//2*builder.cell_mm, builder.rows//2*builder.cell_mm])
    robot_dot, = ax.plot(0, 0, 'ro')
    ax.set_title('Live map building (collecting...)')

    try:
        lidar.start()
        print(f'Collecting scans for {duration_s}s from {port}...')
        t0 = time.time()
        # live loop: collect and update display
        while time.time() - t0 < duration_s:
            raw = lidar.get_raw_scan()
            pts = []
            for q, ang, dist in raw:
                if dist <= 0 or dist > 10000:
                    continue
                a = np.deg2rad(ang)
                x = dist * np.cos(a)
                y = dist * np.sin(a)
                pts.append([x, y])
            if pts:
                builder.update_scan(np.array(pts, dtype=float))
            # smoothing
            builder.logodds *= builder.decay_factor

            # update display
            img_disp = builder.get_display()
            im.set_data(img_disp)
            ax.set_title(f'Live map building (t={int(time.time()-t0)}s)')
            plt.pause(0.05)

    finally:
        try:
            lidar.stop()
        except Exception:
            pass

    # After collection: stabilize and save as before
    occ_counts = builder.occ_count
    occ_mask = occ_counts >= 3
    occ_mask = morphological_close(occ_mask, iters=2)

    img = np.zeros((builder.rows, builder.cols, 3), dtype=float)
    img[:, :, 0] = 0.03
    img[:, :, 1] = 0.06
    img[:, :, 2] = 0.12
    img[occ_mask, 0] = 0.3
    img[occ_mask, 1] = 0.92
    img[occ_mask, 2] = 0.86

    timestamp = int(time.time())
    basename = outname or f'map_{timestamp}'
    png_path = basename + '.png'
    npz_path = basename + '.npz'
    json_path = basename + '.json'

    plt.figure(figsize=(10, 8))
    plt.imshow(img, origin='lower', extent=[-builder.cols//2*builder.cell_mm, builder.cols//2*builder.cell_mm, -builder.rows//2*builder.cell_mm, builder.rows//2*builder.cell_mm])
    plt.plot(0, 0, 'ro')
    plt.title(f'Stable map ({duration_s}s, cell={cell_mm}mm)')
    plt.savefig(png_path)
    print(f'Saved PNG map to {png_path}')

    np.savez_compressed(npz_path, logodds=builder.logodds, occ_count=builder.occ_count, cell_mm=builder.cell_mm)
    print(f'Saved NPZ to {npz_path}')

    metrics = {
        'duration_s': duration_s,
        'cell_mm': cell_mm,
        'port': port,
        'png': png_path,
        'npz': npz_path,
    }
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Saved metadata to {json_path}')

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=30)
    parser.add_argument('--cell', type=int, default=25)
    parser.add_argument('--port', type=str, default=None)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    run(duration_s=args.duration, cell_mm=args.cell, port=args.port, outname=args.out)


if __name__ == '__main__':
    main()
