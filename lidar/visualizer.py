"""
LiDAR data collector - captures scans and outputs report at the end.

Simple terminal-based system without visualization.
Generates random poles and collects LiDAR measurements for analysis.
"""

from __future__ import annotations

from typing import List, Tuple, Dict
import logging
import os
import sys
import random
import time

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


def run_visualizer(lidar: Lidar, duration_sec: int = 30) -> None:
    """
    Collect LiDAR data and output complete report at the end.
    
    Args:
        lidar: Lidar instance.
        duration_sec: Collection duration in seconds (default 30).
    """
    logger = logging.getLogger(__name__)
    
    # Generate random poles
    poles = _generate_random_poles()
    num_poles = len(poles)
    
    logger.info(f"")
    logger.info(f"╔════════════════════════════════════════╗")
    logger.info(f"║     Collecte LiDAR ({duration_sec}s)         ║")
    logger.info(f"╚════════════════════════════════════════╝")
    logger.info(f"")
    logger.info(f"🎯 {num_poles} pôles cibles générés:")
    for name, (x, y) in sorted(poles.items()):
        logger.info(f"   {name:2s}: ({x:7.0f}, {y:7.0f}) mm")
    
    logger.info(f"")
    logger.info(f"⏱️  Démarrage de la collecte... (Ctrl+C pour arrêter)")
    logger.info(f"")
    
    # Collect data
    all_scans: List[List] = []
    start_time = time.time()
    scan_count = 0
    
    try:
        while time.time() - start_time < duration_sec:
            scan = lidar.get_raw_scan()
            if scan:
                all_scans.append(scan)
                scan_count += 1
            time.sleep(0.1)  # Small delay to avoid busy-waiting
    except KeyboardInterrupt:
        logger.info(f"⏹️  Arrêt manuel")
    
    elapsed = time.time() - start_time
    
    if scan_count == 0:
        logger.warning("⚠️  Aucune donnée collectée")
        return
    
    # Collect all measurements into a simple list
    all_measurements = []
    for scan in all_scans:
        for quality, angle_deg, distance_mm in scan:
            all_measurements.append((quality, angle_deg, distance_mm))
    
    # Output final list
    logger.info(f"")
    logger.info(f"╔════════════════════════════════════════╗")
    logger.info(f"║          LISTE COMPLÈTE                ║")
    logger.info(f"╚════════════════════════════════════════╝")
    logger.info(f"")
    logger.info(f"Durée: {elapsed:.1f}s | Scans: {scan_count} | Points: {len(all_measurements)}")
    logger.info(f"")
    logger.info(str(all_measurements))


def main() -> None:
    """Run the LiDAR collector as standalone."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    port = find_lidar_port()
    logger.info(f"Port LiDAR détecté: {port}")

    try:
        with Lidar(port=port) as lidar:
            lidar.start()
            logger.info("✓ LiDAR démarré")
            time.sleep(1)  # Let it stabilize
            
            # Collect for 30 seconds
            run_visualizer(lidar, duration_sec=30)
            
    except KeyboardInterrupt:
        logger.info("⏹️  Arrêt du système")
    except Exception as e:
        logger.error(f"Erreur: {e}", exc_info=True)
    finally:
        logger.info("✓ Système arrêté")


if __name__ == "__main__":
    main()
