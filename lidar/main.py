"""
Simple LiDAR robot exploration with random pole placement.

Demonstrates exploration cartography (Roomba-style) with minimal setup.
"""

import logging
from lidar.lidar import Lidar, find_lidar_port
from lidar.visualizer import run_visualizer


def main():
    """Run LiDAR exploration with random pole placement."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)-8s] %(message)s",
    )
    logger = logging.getLogger(__name__)
    
    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║   LiDAR Robot Exploration & Cartography  ║")
    logger.info("║     Random Pole Placement (3-6 poles)    ║")
    logger.info("╚══════════════════════════════════════════╝")
    
    # Create and run with context manager (automatic cleanup)
    with Lidar(port=find_lidar_port()) as lidar:
        try:
            lidar.start()
            logger.info("✓ LiDAR started")
            logger.info("✓ Starting exploration in 3 seconds...")
            
            # Launch visualization (blocking)
            run_visualizer(lidar)
            
        except KeyboardInterrupt:
            logger.info("⏹ Stopped by user")
        except Exception as e:
            logger.error(f"Error: {e}")
    
    logger.info("✓ Exploration completed")


if __name__ == "__main__":
    main()

