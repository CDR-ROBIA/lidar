"""
Simple LiDAR robot exploration system - no complex localization.

Just reads raw LiDAR scans and tracks robot position for cartography.
Perfect base for adding pole detection/analysis afterwards.
"""

import logging
import os
import threading
import time
from typing import Optional, Tuple, Dict, List

from rplidar import RPLidar
import glob

if __package__ in (None, ""):
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from lidar.state import RobotState
else:
    from .state import RobotState

def find_lidar_port() -> str:
    """Auto-detect a likely LiDAR serial port."""
    candidates = []
    candidates.extend(sorted(glob.glob("/dev/ttyUSB*")))
    candidates.extend(sorted(glob.glob("/dev/ttyACM*")))
    candidates.extend(sorted(glob.glob("/dev/serial/by-id/*")))
    if candidates:
        return candidates[0]
    return "COM5"


class Lidar:
    """
    Simple LiDAR robot for exploration and cartography.
    
    Reads raw LiDAR scans, maintains robot position for map building.
    No complex localization - just basic exploration.
    """
    
    TERRAIN_X_MM = 3000
    TERRAIN_Y_MM = 2000
    DIST_MAX_MM = 5000  # Max distance to accept
    
    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        start_x: float = 1500.0,
        start_y: float = 1000.0,
        start_heading: float = 0.0,
    ) -> None:
        """
        Initialize LiDAR robot.
        
        Args:
            port: Serial port (e.g., /dev/ttyUSB0)
            baudrate: Serial baud rate
            start_x: Initial robot X position (mm)
            start_y: Initial robot Y position (mm)
            start_heading: Initial heading (radians)
        """
        self.port = port
        self.baudrate = baudrate
        
        # Robot state
        self._lock = threading.Lock()
        self._position = (start_x, start_y, start_heading)
        self._scan = []
        self._lidar = None
        
        # Threads
        self._thread_lidar = None
        self._stop_event = threading.Event()
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def start(self) -> None:
        """Start LiDAR acquisition thread."""
        if self._lidar is not None:
            self.logger.warning("LiDAR already running")
            return
        
        try:
            self._lidar = RPLidar(self.port, self.baudrate)
            self._lidar.connect()
            self.logger.info(f"LiDAR connecté sur {self.port} @ {self.baudrate} baud")
        except Exception as e:
            self.logger.error(f"Erreur connexion LiDAR: {e}")
            raise
        
        self._stop_event.clear()
        self._thread_lidar = threading.Thread(target=self._thread_lidar_run, daemon=True)
        self._thread_lidar.start()
    
    def stop(self) -> None:
        """Stop LiDAR and cleanup."""
        self.logger.info("Arrêt du LiDAR...")
        self._stop_event.set()
        
        if self._thread_lidar and self._thread_lidar.is_alive():
            self._thread_lidar.join(timeout=2)
        
        if self._lidar:
            try:
                self._lidar.stop()
                self._lidar.disconnect()
            except Exception as e:
                self.logger.error(f"Erreur arrêt LiDAR: {e}")
            finally:
                self._lidar = None
        
        self.logger.info("LiDAR arrêté")
    
    def _thread_lidar_run(self) -> None:
        """Thread: read scans from RPLidar and display raw data."""
        if not self._lidar:
            return
        
        scan_count = 0
        logging.getLogger(__name__).info("==== ACQUISITIONS LiDAR (RAW) ====")
        
        try:
            for scan in self._lidar.iter_scans():
                if self._stop_event.is_set():
                    break
                
                # Update shared scan data (thread-safe)
                with self._lock:
                    self._scan = list(scan)
                
                # Display first 30 measurements to terminal
                scan_count += 1
                if scan_count % 3 == 0:  # Show every 3rd scan
                    logging.getLogger(__name__).info(f"\n--- Scan #{scan_count} ({len(scan)} points) ---")
                    for idx, (quality, angle_deg, distance_mm) in enumerate(scan[:30]):
                        logging.getLogger(__name__).info(
                            f"[{idx:2d}] quality={quality:3d} angle={angle_deg:7.2f}° dist={distance_mm:7.1f}mm"
                        )
        except Exception as e:
            logging.getLogger(__name__).error(f"Erreur lecture LiDAR: {e}")
    
    def get_pos(self) -> Optional[Tuple[float, float, float]]:
        """Get current robot position (x, y, heading)."""
        with self._lock:
            return self._position
    
    def get_raw_scan(self) -> List:
        """Get last raw scan (list of (quality, angle_deg, distance_mm) tuples)."""
        with self._lock:
            return list(self._scan)
    
    def set_pos(self, x: float, y: float, heading: float) -> None:
        """Manually set robot position (for testing/external odometry)."""
        with self._lock:
            self._position = (x, y, heading)
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

