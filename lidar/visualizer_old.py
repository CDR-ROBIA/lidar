"""
Simple robot cartography visualization - Roomba style.

Real-time matplotlib display of exploration map with random pole placement.
"""

from __future__ import annotations

from typing import List, Tuple, Dict
import logging
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap

if __package__ in (None, ""):
    repo_root = os.path.dirname(os.path.dirname(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from lidar.lidar import Lidar, find_lidar_port
else:
    from .lidar import Lidar, find_lidar_port


def _generate_random_poles(num_poles: int = None) -> Dict[str, Tuple[float, float]]:
    """
    Generate random pole positions.
    
    Args:
        num_poles: Number of poles (3-6). If None, randomly chosen.
    
    Returns:
        Dict with pole names and positions {name: (x, y) in mm}.
    """
    if num_poles is None:
        num_poles = random.randint(3, 6)
    
    num_poles = max(3, min(6, num_poles))  # Clamp to 3-6
    
    # Generate random positions avoiding edges
    poles = {}
    pole_names = [f"P{i+1}" for i in range(num_poles)]
    
    for name in pole_names:
        while True:
            x = random.uniform(300, 2700)
            y = random.uniform(300, 1700)
            # Avoid clustering
            too_close = any(
                ((x - px)**2 + (y - py)**2) < 300**2
                for px, py in poles.values()
            )
            if not too_close:
                poles[name] = (x, y)
                break
    
    return poles


def run_visualizer(lidar: Lidar) -> None:
    """
    Display real-time cartography (Roomba-style exploration).
    
    Uses matplotlib with blit=True optimization.
    Generates random poles at startup for exploration and analysis.
    
    Args:
        lidar: Lidar instance to visualize.
    """
    # Generate random poles
    poles = _generate_random_poles()
    num_poles = len(poles)
    
    logging.getLogger(__name__).info(f"🎯 {num_poles} pôles générés aléatoirement:")
    for name, (x, y) in sorted(poles.items()):
        logging.getLogger(__name__).info(f"  {name}: ({x:.0f}, {y:.0f}) mm")
        # Robot position: fixed for cartography testing
        robot_x_fixed = 1500.0
        robot_y_fixed = 1000.0
        heading_fixed = 0.0
    
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor("#000000")
    fig.patch.set_facecolor("#000000")

    # Vue locale centrée sur le robot (plus zoomée)
    view_range = 2000  # 2m x 2m autour du robot
    ax.set_xlim(robot_x_fixed - view_range/2, robot_x_fixed + view_range/2)
    ax.set_ylim(robot_y_fixed - view_range/2, robot_y_fixed + view_range/2)
    ax.set_aspect("equal")
    ax.grid(True, color="#222222", linewidth=0.3, alpha=0.5)
    ax.tick_params(colors="#666666")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    # ===== GRILLE D'OCCUPATION (Style aspirateur robot) =====
    grid_size = 100  # mm
    grid_cols = int(3000 / grid_size)  # 30
    grid_rows = int(2000 / grid_size)  # 20
    occupancy_grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    
    colors_map = ["#0a0a0a", "#0d3d0d", "#1a661a", "#33cc33", "#99ff99"]
    cmap = LinearSegmentedColormap.from_list("exploration", colors_map)
    
    im = ax.imshow(
        occupancy_grid,
        cmap=cmap,
        extent=[0, 3000, 0, 2000],
        origin="lower",
        aspect="auto",
        alpha=0.4,
        vmin=0,
        vmax=100,
    )

    scatter_scan = ax.scatter([], [], s=3, color="#00aaff", alpha=0.6, label="scan (temps réel)")
    scatter_map = ax.scatter([], [], s=0.5, color="#00ff88", alpha=0.15, label="carto (points)")
    scatter_poles = ax.scatter([], [], s=150, color="#ff4444", marker="^", zorder=5, label=f"pôles ({num_poles})")
    scatter_robot = ax.scatter([], [], s=250, color="#ffff00", marker="o", zorder=6, edgecolors="white", linewidths=2, label="robot (fixe)")

    text_title = ax.text(
        0.5,
        0.99,
        "",
        transform=ax.transAxes,
        color="#00ff88",
        ha="center",
        fontsize=14,
        fontweight="bold",
        verticalalignment="top",
    )
    text_info = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        color="#00ff88",
        fontsize=9,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="#0d0d0d", alpha=0.9, edgecolor="#00ff88"),
    )

    # Pole positions
    pole_xy = np.array(list(poles.values()))
    mapped_points: List[Tuple[float, float]] = []
    
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

    def update_frame(frame):
        nonlocal occupancy_grid
        artists = []

        # Get raw scan from LiDAR
        scan = lidar.get_raw_scan()
        
        # Use fixed position for cartography
        robot_x, robot_y, heading = robot_x_fixed, robot_y_fixed, heading_fixed
        lidar.set_pos(robot_x, robot_y, heading)
        
        scatter_robot.set_offsets([[robot_x, robot_y]])
        artists.append(scatter_robot)

        if len(scan) > 0:
            abs_xs, abs_ys = _get_scan_points_absolute(scan, robot_x, robot_y, heading)
            
            # ===== MISE À JOUR DE LA GRILLE D'OCCUPATION =====
            for x, y in zip(abs_xs, abs_ys):
                if 0 <= x < 3000 and 0 <= y < 2000:
                    col = int(x / grid_size)
                    row = int(y / grid_size)
                    if 0 <= col < grid_cols and 0 <= row < grid_rows:
                        occupancy_grid[row, col] = min(100, occupancy_grid[row, col] + 2)
            
            for x, y in zip(abs_xs, abs_ys):
                mapped_points.append((x, y))
            if len(mapped_points) > 100000:
                mapped_points.pop(0)

        scatter_poles.set_offsets(pole_xy)
        artists.append(scatter_poles)

        # ===== MÀJ HEATMAP D'EXPLORATION =====
        im.set_data(occupancy_grid)
        artists.append(im)

        if mapped_points:
            map_xy = np.array(mapped_points)
            scatter_map.set_offsets(map_xy)
        else:
            scatter_map.set_offsets(np.empty((0, 2)))
        artists.append(scatter_map)

        if len(scan) > 0:
            xs, ys = _get_scan_points_absolute(scan, robot_x, robot_y, heading)
            scatter_scan.set_offsets(np.c_[xs, ys])
        else:
            scatter_scan.set_offsets(np.empty((0, 2)))
        artists.append(scatter_scan)

        heading_deg = np.degrees(heading)
        coverage_pct = int(np.count_nonzero(occupancy_grid > 0) / (grid_rows * grid_cols) * 100)
        text_title.set_text(
            f"X={int(robot_x):4d}mm  Y={int(robot_y):4d}mm  Heading={heading_deg:6.1f}°  Couverture={coverage_pct:3d}%"
        )
        text_title.set_color("#00ff88")

        artists.append(text_title)

        info_text = _build_info_text(lidar)
        text_info.set_text(info_text)
        artists.append(text_info)

        return artists

    animation.FuncAnimation(
        fig,
        update_frame,
        interval=300,
        blit=True,
        cache_frame_data=False,
    )

    try:
        logger = logging.getLogger(__name__)
        logger.info("Visualizer ouvert - Appuyez sur Ctrl+C pour arrêter")
        plt.show()
    except KeyboardInterrupt:
        logger = logging.getLogger(__name__)
        logger.info("Arrêt en cours...")
    finally:
        plt.close(fig)



def _get_scan_points_absolute(
    scan: List,
    robot_x: float,
    robot_y: float,
    heading: float,
) -> Tuple[List[float], List[float]]:
    """Convert scan points from LiDAR polar coordinates to absolute cartesian."""
    dist_max = int(np.sqrt(3000**2 + 2000**2))
    xs, ys = [], []

    for i, measurement in enumerate(scan):
        if i % 3 != 0:
            continue

        angle_lidar_deg = measurement[1]
        distance = measurement[2]

        if distance <= 0 or distance > dist_max:
            continue

        angle_absolu = heading - np.radians(angle_lidar_deg)
        px = robot_x + distance * np.cos(angle_absolu)
        py = robot_y + distance * np.sin(angle_absolu)

        xs.append(px)
        ys.append(py)

    return xs, ys


def _get_scan_points_local(scan: List) -> Tuple[List[float], List[float]]:
    """Convert scan points to the LiDAR-local XY frame."""
    xs, ys = [], []

    for i, measurement in enumerate(scan):
        if i % 3 != 0:
            continue

        angle_lidar_deg = measurement[1]
        distance = measurement[2]

        if distance <= 0:
            continue

        angle_rad = np.radians(angle_lidar_deg)
        xs.append(distance * np.cos(angle_rad))
        ys.append(distance * np.sin(angle_rad))

    return xs, ys


def _build_info_text(lidar: Lidar) -> str:
    """Build information text for top-left corner."""
    pos = lidar.get_pos()
    scan = lidar.get_raw_scan()

    if pos is None:
        return "Position: initializing...\nStatus: waiting for data"

    robot_x, robot_y, heading = pos
    heading_deg = np.degrees(heading)

    info = f"Position: ({int(robot_x)}, {int(robot_y)}) mm\n"
    info += f"Heading: {heading_deg:.1f}°\n"
    info += f"Scan points: {len(scan)}\n"

    return info


def main() -> None:
    """Run the visualizer as a standalone USB LiDAR viewer."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    port = find_lidar_port()
    logger.info(f"Using LiDAR port: {port}")

    try:
        with Lidar(port=port) as lidar:
            lidar.start()
            logger.info("LiDAR started, opening visualizer...")
            logger.info("Appuyez sur Ctrl+C pour arrêter le système")
            run_visualizer(lidar)
    except KeyboardInterrupt:
        logger.info("Arrêt du système LiDAR...")
    except Exception as e:
        logger.error(f"Erreur fatale: {e}", exc_info=True)
    finally:
        logger.info("Système arrêté")


if __name__ == "__main__":
    main()
