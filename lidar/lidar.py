"""
Lidar robot localization system.

Main class encapsulating the complete LiDAR-based localization pipeline.
All business logic from the original monolithic script, refactored into OOP.
"""

import logging
import threading
import queue
import time
from typing import Optional, Tuple, Dict, List
import numpy as np
from rplidar import RPLidar

from detection_poteaux import PoleDetector
from .state import RobotState
from .localization import trilaterate, estimate_heading_from_pole, average_headings


class Lidar:
    """
    Robot localization system using LiDAR and trilateration.
    
    This class manages:
    - Hardware connection and scan acquisition
    - Pole detection via DBSCAN clustering
    - Position/heading estimation via trilateration
    - Thread-safe state management
    - Graceful lifecycle (start/stop)
    
    Thread model:
    - _thread_lidar: reads scans from RPLidar, pushes to queue (non-blocking)
    - _thread_detection: consumes queue, runs DBSCAN, updates state
    - Main thread: can call get_pos() anytime (thread-safe read)
    
    Context manager protocol is supported for automatic cleanup.
    """
    
    # Arena and pole configuration constants
    TERRAIN_X_MM = 3000
    TERRAIN_Y_MM = 2000
    DIST_MAX_MM = int(np.sqrt(TERRAIN_X_MM**2 + TERRAIN_Y_MM**2))  # ~3606
    DIST_P2_P3_MM = 2000  # Distance between poles P2 and P3 (arena height)
    
    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        terrain_x: int = 3000,
        terrain_y: int = 2000,
        poles: Optional[Dict[str, Tuple[float, float]]] = None,
        buffer_size: int = 4,
        tolerance_dist: float = 150.0,
        tolerance_angle: float = 10.0,
    ) -> None:
        """
        Initialize LiDAR localization system.
        
        All parameters become instance attributes, overriding class constants.
        
        Args:
            port: Serial port for RPLidar (e.g., "COM5" or "/dev/ttyUSB0").
            baudrate: Serial communication baudrate (default 115200).
            terrain_x: Arena width in mm (default 3000).
            terrain_y: Arena height in mm (default 2000).
            poles: Dict of pole positions {name: (x, y) in mm}.
                   Default: {"P1": (0, 0), "P2": (3000, 0), "P3": (3000, 2000)}.
            buffer_size: Number of scans to buffer in pole detector (default 4).
            tolerance_dist: Distance tolerance for validation (mm).
            tolerance_angle: Angular tolerance for validation (degrees).
        """
        self._logger = logging.getLogger(__name__)
        
        # Configuration
        self._port = port
        self._baudrate = baudrate
        self._terrain_x = terrain_x
        self._terrain_y = terrain_y
        self._tolerance_dist = tolerance_dist
        self._tolerance_angle = tolerance_angle
        
        # Poles (convert to numpy arrays for math operations)
        if poles is None:
            poles = {
                "P1": (0, 0),
                "P2": (3000, 0),
                "P3": (3000, 2000),
            }
        self._poles = {
            name: np.array(pos, dtype=float)
            for name, pos in poles.items()
        }
        
        # Hardware
        self._lidar: Optional[RPLidar] = None
        self._scan_iterator = None
        
        # Pole detection
        self._detector = PoleDetector(buffer_size=buffer_size)
        
        # Threading
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._scan_queue: queue.Queue = queue.Queue(maxsize=10)
        
        # State (protected by _lock)
        self._state = RobotState(status="not started")
        
        # Threads (created at start())
        self._thread_lidar: Optional[threading.Thread] = None
        self._thread_detection: Optional[threading.Thread] = None
        
        self._logger.info(
            f"Initialized Lidar system (port={port}, terrain={terrain_x}x{terrain_y}mm, "
            f"buffer={buffer_size})"
        )
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def start(self) -> None:
        """
        Start LiDAR hardware and launch threads.
        
        Raises:
            RuntimeError: If hardware connection fails.
        """
        self._logger.info("Starting LiDAR system...")
        
        # Connect hardware
        try:
            self._lidar = RPLidar(self._port, baudrate=self._baudrate)
            self._lidar.stop()
            self._lidar.stop_motor()
            time.sleep(1)
            self._lidar.start_motor()
            time.sleep(1)
            self._scan_iterator = self._lidar.iter_scans(max_buf_meas=500)
            self._logger.info(f"LiDAR connected on {self._port}")
        except Exception as e:
            self._logger.error(f"Failed to connect LiDAR: {e}")
            raise RuntimeError(f"LiDAR connection failed on {self._port}") from e
        
        # Clear stop event (in case of restart)
        self._stop_event.clear()
        
        # Start threads
        self._thread_lidar = threading.Thread(target=self._thread_lidar_run, daemon=True)
        self._thread_detection = threading.Thread(target=self._thread_detection_run, daemon=True)
        
        self._thread_lidar.start()
        self._thread_detection.start()
        
        self._logger.info("LiDAR threads started")
    
    def stop(self) -> None:
        """Stop LiDAR hardware and threads gracefully."""
        self._logger.info("Stopping LiDAR system...")
        
        self._stop_event.set()
        
        # Wait for threads with timeout
        if self._thread_lidar:
            self._thread_lidar.join(timeout=2.0)
        if self._thread_detection:
            self._thread_detection.join(timeout=2.0)
        
        # Cleanup hardware
        if self._lidar:
            try:
                self._lidar.stop()
                self._lidar.stop_motor()
                self._lidar.disconnect()
                self._logger.info("LiDAR disconnected")
            except Exception as e:
                self._logger.error(f"Error during LiDAR shutdown: {e}")
    
    def get_pos(self) -> Optional[Tuple[float, float, float]]:
        """
        Get current robot position and heading.
        
        Returns:
            (x_mm, y_mm, heading_rad) if a valid position is available,
            None otherwise.
            
        Thread-safe: returns a snapshot without blocking.
        """
        with self._lock:
            if self._state.is_valid():
                return self._state.get_position()
            return None
    
    def is_ready(self) -> bool:
        """Check if at least one valid position has been computed."""
        with self._lock:
            return self._state.is_valid()
    
    def get_raw_scan(self) -> List:
        """Get a copy of the last raw LiDAR scan."""
        with self._lock:
            return list(self._state.scan)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, *args):
        """Context manager exit: ensure cleanup."""
        self.stop()
    
    # =========================================================================
    # PRIVATE THREADS
    # =========================================================================
    
    def _thread_lidar_run(self) -> None:
        """
        LiDAR reader thread: acquire scans and push to queue.
        
        Runs at maximum speed (scan acquisition rate ~9 Hz).
        Recovers from errors: after 3 consecutive failures, stops system.
        """
        consecutive_failures = 0
        
        while not self._stop_event.is_set():
            try:
                scan = next(self._scan_iterator, None)
                
                if scan is None:
                    consecutive_failures += 1
                    self._logger.warning(
                        f"Failed to acquire scan (attempt {consecutive_failures}/3)"
                    )
                    
                    if consecutive_failures >= 3:
                        self._logger.error("LiDAR error threshold reached, stopping")
                        self._stop_event.set()
                        break
                    
                    time.sleep(0.1)
                    continue
                
                consecutive_failures = 0
                
                # Update state snapshot
                with self._lock:
                    self._state.scan = scan
                    self._state.num_scan += 1
                
                # Push to detection queue
                try:
                    self._scan_queue.put_nowait(scan)
                except queue.Full:
                    pass  # Drop oldest scan if queue is full
                
            except Exception as e:
                self._logger.error(f"LiDAR read exception: {e}")
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    self._stop_event.set()
                time.sleep(0.1)
    
    def _thread_detection_run(self) -> None:
        """
        Detection thread: DBSCAN pole detection + trilateration.
        
        Consumes scans from queue, detects poles, computes position/heading,
        updates shared state.
        """
        while not self._stop_event.is_set():
            try:
                # Wait for next scan (with timeout to check stop_event)
                try:
                    scan = self._scan_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Add scan to detector buffer
                self._detector.add_scan(scan)
                
                if not self._detector.ready:
                    continue
                
                # Detect poles
                poles_dict = self._detector.detect()
                if poles_dict is None:
                    with self._lock:
                        self._state.status = "pole detection failed"
                    self._logger.debug("No poles detected in current buffer")
                    continue
                
                # Extract measurements
                try:
                    dist_p1_mm = poles_dict["P1"]["distance"]
                    angle_p1_deg = poles_dict["P1"]["angle"]
                    
                    dist_p2_mm = poles_dict["P2"]["distance"]
                    angle_p2_deg = poles_dict["P2"]["angle"]
                    
                    dist_p3_mm = poles_dict["P3"]["distance"]
                    angle_p3_deg = poles_dict["P3"]["angle"]
                except KeyError as e:
                    self._logger.warning(f"Incomplete pole detection: {e}")
                    continue
                
                # Validate distances
                if not self._validate_distances(dist_p1_mm, dist_p2_mm, dist_p3_mm):
                    with self._lock:
                        self._state.status = "invalid distances"
                    continue
                
                # Trilaterate
                try:
                    robot_position = trilaterate(
                        self._poles["P1"], dist_p1_mm,
                        self._poles["P2"], dist_p2_mm,
                        self._poles["P3"], dist_p3_mm,
                    )
                except ValueError:
                    with self._lock:
                        self._state.status = "distance validation failed"
                    continue
                
                if robot_position is None:
                    with self._lock:
                        self._state.status = "trilateration impossible (collinear poles)"
                    self._logger.debug("Trilateration failed (singular system)")
                    continue
                
                # Validate position bounds
                if not self._is_in_bounds(robot_position):
                    with self._lock:
                        self._state.status = "position out of bounds"
                    self._logger.debug(
                        f"Position out of bounds: ({robot_position[0]:.0f}, "
                        f"{robot_position[1]:.0f})"
                    )
                    continue
                
                # Estimate heading from each pole
                heading_p1 = estimate_heading_from_pole(
                    robot_position, self._poles["P1"], angle_p1_deg
                )
                heading_p2 = estimate_heading_from_pole(
                    robot_position, self._poles["P2"], angle_p2_deg
                )
                heading_p3 = estimate_heading_from_pole(
                    robot_position, self._poles["P3"], angle_p3_deg
                )
                
                # Average headings (circular mean)
                heading = average_headings(heading_p1, heading_p2, heading_p3)
                
                # Update state
                self._update_state(
                    robot_position[0],
                    robot_position[1],
                    heading,
                    dist_p1_mm,
                    dist_p2_mm,
                    dist_p3_mm,
                    angle_p1_deg,
                    angle_p2_deg,
                    angle_p3_deg,
                )
                
                self._logger.debug(
                    f"Position: X={robot_position[0]:.1f}mm Y={robot_position[1]:.1f}mm "
                    f"Heading={np.degrees(heading):.1f}°"
                )
                
            except Exception as e:
                self._logger.error(f"Detection thread exception: {e}", exc_info=True)
    
    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================
    
    def _validate_distances(self, r1: Optional[float], r2: Optional[float], r3: Optional[float]) -> bool:
        """
        Validate distance measurements.
        
        Args:
            r1, r2, r3: Measured distances in mm.
        
        Returns:
            bool: True if all distances are valid (positive, within max range, not None).
        """
        return (
            r1 is not None and r2 is not None and r3 is not None and
            r1 > 0 and r2 > 0 and r3 > 0 and
            r1 < self.DIST_MAX_MM and r2 < self.DIST_MAX_MM and r3 < self.DIST_MAX_MM
        )
    
    def _is_in_bounds(self, robot_position: np.ndarray) -> bool:
        """Check if position is within arena bounds."""
        return (
            0 <= robot_position[0] <= self._terrain_x and
            0 <= robot_position[1] <= self._terrain_y
        )
    
    def _update_state(
        self,
        robot_x: float,
        robot_y: float,
        heading: float,
        dist_p1_mm: float,
        dist_p2_mm: float,
        dist_p3_mm: float,
        angle_p1_deg: float,
        angle_p2_deg: float,
        angle_p3_deg: float,
    ) -> None:
        """Update shared state (thread-safe write)."""
        with self._lock:
            self._state = RobotState(
                robot_x=robot_x,
                robot_y=robot_y,
                heading=heading,
                scan=self._state.scan,  # Preserve scan
                dist_p1_mm=dist_p1_mm,
                dist_p2_mm=dist_p2_mm,
                dist_p3_mm=dist_p3_mm,
                angle_p1_deg=angle_p1_deg,
                angle_p2_deg=angle_p2_deg,
                angle_p3_deg=angle_p3_deg,
                num_scan=self._state.num_scan,
                status="OK",
            )
