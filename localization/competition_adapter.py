"""
Competition adapter: integrates with existing `lidar.Lidar` (no changes to it),
performs initial pillar calibration (3-6 pillars), then switches to opponent
detection and logs clean structured data.
"""
from __future__ import annotations
import time
import json
from typing import List, Dict, Tuple, Optional
import numpy as np
import os

from scan_processor import ScanProcessor
from pillar_detector import detect_pillars, filter_pillars_by_consistency

try:
    from lidar.lidar import Lidar, find_lidar_port
except Exception:
    Lidar = None
    find_lidar_port = None
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from lidar.lidar import Lidar, find_lidar_port  # type: ignore
    except Exception:
        Lidar = None
        find_lidar_port = None


class CompetitionAdapter:
    def __init__(self, port: str = None, baudrate: int = 115200):
        self.port = port or (find_lidar_port() if find_lidar_port else None)
        self.baudrate = baudrate
        self.lidar = None
        self.sp = ScanProcessor()

        # history of pillar detections (per scan)
        self.pillar_history: List[List[Dict]] = []
        self.confirmed_pillars: List[Dict] = []

        # opponent tracking state
        self.opponent_history: List[Dict] = []
        self.last_result: Dict = {}

    def start_lidar(self):
        if Lidar is None:
            raise RuntimeError('LiDAR module not available')
        self.lidar = Lidar(self.port)
        self.lidar.start()

    def stop_lidar(self):
        if self.lidar:
            try:
                self.lidar.stop()
            except Exception:
                pass

    def capture_scan_points(self) -> np.ndarray:
        raw = self.lidar.get_raw_scan()
        pts = []
        for q, ang, dist in raw:
            if dist <= 0:
                continue
            a = np.radians(ang)
            x = dist * np.cos(a)
            y = dist * np.sin(a)
            pts.append([x, y])
        return np.array(pts, dtype=float)

    def calibrate_pillars(self,
                          min_required: int = 3,
                          max_required: int = 6,
                          scans: int = 30,
                          timeout_s: float = 10.0) -> List[Dict]:
        """Collect scans and detect consistent pillars.

        Returns up to `max_required` pillars; raises ValueError if fewer than
        `min_required` are confirmed.
        """
        self.pillar_history = []
        t0 = time.time()
        scan_count = 0
        while (scan_count < scans) and (time.time() - t0 < timeout_s):
            pts = self.capture_scan_points()
            processed = self.sp.process_scan([(10, float(np.degrees(np.arctan2(y, x))), float(np.hypot(x, y))) for x, y in pts])
            dets = detect_pillars(processed, min_points=6, dist_thresh=80.0)
            self.pillar_history.append(dets)
            scan_count += 1
            time.sleep(0.05)

        confirmed = filter_pillars_by_consistency(self.pillar_history, min_observations=3, position_tol_mm=200.0)
        self.confirmed_pillars = confirmed[:max_required]
        if len(self.confirmed_pillars) < min_required:
            raise ValueError(
                f"Calibration failed: detected {len(self.confirmed_pillars)} pillars, "
                f"need at least {min_required}."
            )
        return self.confirmed_pillars

    def _is_near_confirmed_pillar(self, x: float, y: float, tol_mm: float = 250.0) -> bool:
        for p in self.confirmed_pillars:
            px, py = p['centroid']
            if np.hypot(x - px, y - py) <= tol_mm:
                return True
        return False

    def detect_opponent(self, scans_to_watch: int = 5, movement_threshold: float = 200.0) -> List[Dict]:
        """Simple opponent detection: look for moving clusters across recent scans.

        Returns list of candidate moving objects with last seen position.
        """
        recent: List[List[Dict]] = []
        for _ in range(scans_to_watch):
            pts = self.capture_scan_points()
            processed = self.sp.process_scan([(10, float(np.degrees(np.arctan2(y, x))), float(np.hypot(x, y))) for x, y in pts])
            dets = detect_pillars(processed, min_points=4, dist_thresh=120.0)
            recent.append(dets)
            time.sleep(0.05)

        # flatten by cluster centroids and detect clusters that move between scans
        # build centroid lists per scan
        centroids_per_scan = [[(d['centroid'][0], d['centroid'][1]) for d in s] for s in recent]
        # match centroids by nearest neighbor and compute displacement
        moving_candidates: List[Dict] = []
        if not centroids_per_scan:
            return []

        base = centroids_per_scan[0]
        for c in base:
            xs = [c]
            for s in centroids_per_scan[1:]:
                if not s:
                    xs.append(None)
                    continue
                # find nearest
                dists = [np.hypot(c[0]-q[0], c[1]-q[1]) for q in s]
                idx = int(np.argmin(dists))
                if dists[idx] < 300.0:
                    xs.append(s[idx])
                else:
                    xs.append(None)

            # compute displacement if present
            valid = [p for p in xs if p is not None]
            if len(valid) >= 2:
                # compare first and last
                dx = valid[-1][0] - valid[0][0]
                dy = valid[-1][1] - valid[0][1]
                dist = np.hypot(dx, dy)
                if dist >= movement_threshold:
                    end_x, end_y = valid[-1]
                    if self._is_near_confirmed_pillar(end_x, end_y):
                        continue
                    moving_candidates.append({
                        'start': valid[0],
                        'end': valid[-1],
                        'displacement_mm': float(dist)
                    })

        moving_candidates.sort(key=lambda m: -m['displacement_mm'])
        if moving_candidates:
            best = moving_candidates[0]
            self.opponent_history.append({
                'timestamp': time.time(),
                'x_mm': best['end'][0],
                'y_mm': best['end'][1],
                'displacement_mm': best['displacement_mm']
            })

        return moving_candidates

    def run_cycle(self,
                  min_pillars: int = 3,
                  max_pillars: int = 6,
                  calibration_scans: int = 40,
                  watch_windows: int = 10) -> Dict:
        """Complete cycle for competition start: calibrate pillars then detect opponent."""
        pillars = self.calibrate_pillars(
            min_required=min_pillars,
            max_required=max_pillars,
            scans=calibration_scans,
            timeout_s=20.0,
        )

        opponent_candidates = self.detect_opponent(scans_to_watch=watch_windows, movement_threshold=150.0)
        best_opponent: Optional[Dict] = opponent_candidates[0] if opponent_candidates else None

        self.last_result = {
            'timestamp': int(time.time()),
            'pillars_count': len(pillars),
            'pillars': pillars,
            'opponent_detected': best_opponent is not None,
            'best_opponent': best_opponent,
            'opponent_candidates': opponent_candidates,
        }
        return self.last_result

    def export_calibration(self, out_path: str):
        data = self.last_result or {
            'pillars': self.confirmed_pillars,
            'timestamp': int(time.time())
        }
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)
