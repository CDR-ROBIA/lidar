"""
Example usage of Lidar system.

Minimal usage demonstrating context manager and main loop.
"""

import logging
import time
import numpy as np

from lidar.lidar import Lidar
from lidar.visualizer import run_visualizer


def main():
    """Run LiDAR localization system with visualization."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("LiDAR Robot Localization System - Refactored OOP Version")
    logger.info("=" * 80)
    
    # Create system with context manager (automatic cleanup)
    with Lidar(port="COM5") as lidar:
        try:
            lidar.start()
            logger.info("LiDAR started, waiting for first position...")
            
            # Wait for system to be ready
            start_time = time.time()
            while not lidar.is_ready() and time.time() - start_time < 10:
                time.sleep(0.5)
            
            if not lidar.is_ready():
                logger.warning("LiDAR not ready after 10 seconds")
            else:
                logger.info("LiDAR ready!")
            
            # Launch visualization (blocking call)
            run_visualizer(lidar)
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
    
    logger.info("LiDAR system stopped")
    logger.info("=" * 80)


def example_polling():
    """Example: polling loop without visualization."""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    with Lidar(port="COM5") as lidar:
        lidar.start()
        
        # Poll position for 30 seconds
        start = time.time()
        while time.time() - start < 30:
            pos = lidar.get_pos()
            
            if pos:
                x, y, heading = pos
                heading_deg = np.degrees(heading)
                logger.info(f"Position: ({x:.1f}, {y:.1f}) mm, Heading: {heading_deg:.1f}°")
            else:
                logger.info("Position not available yet")
            
            time.sleep(1)


if __name__ == "__main__":
    main()
    # Uncomment to test polling mode:
    # example_polling()
