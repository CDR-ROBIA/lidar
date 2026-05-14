"""
Simple and clean competition pipeline.

Goals:
1) Initial calibration: detect and lock 3..6 pillars.
2) Runtime: detect moving opponent while ignoring fixed pillars.

This module does not modify existing `lidar/*` files.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import json
import math
import os
import time
import glob

import numpy as np

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


@dataclass
class Pillar:
    x_mm: float
    y_mm: float
    radius_mm: float
    observations: int


@dataclass
class Opponent:
    x_mm: float
    y_mm: float
    displacement_mm: float
    confidence: float


class SimpleCompetitionPipeline:
    def __init__(self, port: Optional[str] = None, baudrate: int = 115200):
        self.port = port or (find_lidar_port() if find_lidar_port else None)
        self.baudrate = baudrate
        self.lidar = None
        self.pillars: List[Pillar] = []
        self.last_points_count: int = 0
        self.empty_scan_streak: int = 0
        self.started_at: int = 0

    @staticmethod
    def _list_candidate_ports() -> List[str]:
        """Return likely serial ports for Raspberry/Linux in preferred order."""
        candidates: List[str] = []
        # Most stable on Linux/Raspberry: by-id symlink
        candidates.extend(sorted(glob.glob("/dev/serial/by-id/*")))
        candidates.extend(sorted(glob.glob("/dev/ttyUSB*")))
        candidates.extend(sorted(glob.glob("/dev/ttyACM*")))
        # Deduplicate while preserving order
        seen = set()
        ordered = []
        for p in candidates:
            if p not in seen:
                seen.add(p)
                ordered.append(p)
        return ordered

    def auto_detect_port(self, timeout_s: float = 20.0, retry_interval_s: float = 1.0) -> Optional[str]:
        """Wait for a LiDAR serial port to appear and return it.

        Useful on Raspberry when USB devices enumerate after app startup.
        """
        t0 = time.time()
        while time.time() - t0 <= timeout_s:
            # Try project helper first if available
            if find_lidar_port is not None:
                try:
                    p = find_lidar_port()
                    if p and os.path.exists(p):
                        return p
                except Exception:
                    pass

            # Fallback direct scan
            ports = self._list_candidate_ports()
            if ports:
                return ports[0]

            time.sleep(retry_interval_s)
        return None

    def start(self, auto_detect: bool = True, detect_timeout_s: float = 20.0) -> None:
        if Lidar is None:
            raise RuntimeError("LiDAR module unavailable in this environment")

        # Build candidate list: explicit port first (if provided), then auto-detected candidates
        candidates: List[str] = []
        if self.port:
            candidates.append(self.port)

        if auto_detect:
            detected = self.auto_detect_port(timeout_s=detect_timeout_s)
            if detected and detected not in candidates:
                candidates.append(detected)
            for p in self._list_candidate_ports():
                if p not in candidates:
                    candidates.append(p)

        if not candidates:
            raise RuntimeError("No LiDAR serial port detected")

        last_error = None
        for p in candidates:
            try:
                self.lidar = Lidar(p, baudrate=self.baudrate)
                self.lidar.start()
                self.port = p
                self.started_at = int(time.time())
                return
            except Exception as e:
                last_error = e
                self.lidar = None

        raise RuntimeError(f"Unable to start LiDAR on candidates {candidates}: {last_error}")

    def stop(self) -> None:
        if self.lidar:
            try:
                self.lidar.stop()
            except Exception:
                pass

    def _scan_to_points(self) -> np.ndarray:
        raw = self.lidar.get_raw_scan()
        points = []
        for quality, angle_deg, distance_mm in raw:
            if quality < 5:
                continue
            if distance_mm <= 0 or distance_mm > 6000:
                continue
            a = math.radians(angle_deg)
            x = distance_mm * math.cos(a)
            y = distance_mm * math.sin(a)
            points.append((x, y))
        if not points:
            self.last_points_count = 0
            self.empty_scan_streak += 1
            return np.empty((0, 2), dtype=float)
        self.last_points_count = len(points)
        self.empty_scan_streak = 0
        return np.array(points, dtype=float)

    def _cluster_points(self, points: np.ndarray, eps_mm: float = 120.0, min_pts: int = 5) -> List[np.ndarray]:
        if len(points) == 0:
            return []
        n = len(points)
        used = np.zeros(n, dtype=bool)
        clusters: List[np.ndarray] = []

        for i in range(n):
            if used[i]:
                continue
            used[i] = True
            stack = [i]
            members = []
            while stack:
                idx = stack.pop()
                members.append(idx)
                d2 = np.sum((points - points[idx]) ** 2, axis=1)
                neigh = np.where((d2 <= eps_mm ** 2) & (~used))[0]
                for j in neigh:
                    used[j] = True
                    stack.append(j)
            if len(members) >= min_pts:
                clusters.append(points[members])
        return clusters

    def _detect_round_objects(self, points: np.ndarray) -> List[Dict]:
        clusters = self._cluster_points(points, eps_mm=120.0, min_pts=5)
        objects: List[Dict] = []
        for c in clusters:
            center = c.mean(axis=0)
            dist = np.linalg.norm(c - center, axis=1)
            radius = float(np.percentile(dist, 90))
            if 40.0 <= radius <= 350.0:
                objects.append({
                    "x_mm": float(center[0]),
                    "y_mm": float(center[1]),
                    "radius_mm": radius,
                    "points": int(len(c)),
                })
        objects.sort(key=lambda o: o["x_mm"] ** 2 + o["y_mm"] ** 2)
        return objects

    def calibrate_pillars(self, min_pillars: int = 3, max_pillars: int = 6, scans: int = 40) -> List[Pillar]:
        history: List[List[Dict]] = []
        valid_scans = 0
        for _ in range(scans):
            pts = self._scan_to_points()
            if len(pts) == 0:
                time.sleep(0.05)
                continue
            history.append(self._detect_round_objects(pts))
            valid_scans += 1
            time.sleep(0.05)

        if valid_scans < max(10, scans // 4):
            raise ValueError(f"Calibration aborted: insufficient valid scans ({valid_scans}/{scans})")

        all_points: List[Tuple[float, float]] = []
        for dets in history:
            for d in dets:
                all_points.append((d["x_mm"], d["y_mm"]))

        if not all_points:
            raise ValueError("No pillar candidates found during calibration")

        pts = np.array(all_points, dtype=float)
        groups = self._cluster_points(pts, eps_mm=220.0, min_pts=3)
        pillars: List[Pillar] = []
        for g in groups:
            center = g.mean(axis=0)
            r = np.linalg.norm(g - center, axis=1)
            pillars.append(Pillar(
                x_mm=float(center[0]),
                y_mm=float(center[1]),
                radius_mm=float(np.percentile(r, 80)),
                observations=int(len(g)),
            ))

        pillars.sort(key=lambda p: (-p.observations, p.x_mm ** 2 + p.y_mm ** 2))
        pillars = pillars[:max_pillars]
        if len(pillars) < min_pillars:
            raise ValueError(f"Calibration failed: found {len(pillars)} pillars (< {min_pillars})")

        self.pillars = pillars
        return pillars

    def _is_near_pillar(self, x_mm: float, y_mm: float, tol_mm: float = 260.0) -> bool:
        for p in self.pillars:
            if math.hypot(x_mm - p.x_mm, y_mm - p.y_mm) <= tol_mm:
                return True
        return False

    def detect_opponent(self, window_scans: int = 8, movement_threshold_mm: float = 150.0) -> Optional[Opponent]:
        frames: List[List[Dict]] = []
        valid_windows = 0
        for _ in range(window_scans):
            pts = self._scan_to_points()
            if len(pts) == 0:
                time.sleep(0.05)
                continue
            objs = self._detect_round_objects(pts)
            objs = [o for o in objs if not self._is_near_pillar(o["x_mm"], o["y_mm"])]
            frames.append(objs)
            valid_windows += 1
            time.sleep(0.05)

        if valid_windows < 2 or not frames or not frames[0]:
            return None

        base = frames[0]
        best = None
        best_disp = -1.0
        for b in base:
            bx, by = b["x_mm"], b["y_mm"]
            target = None
            for f in reversed(frames[1:]):
                if not f:
                    continue
                dists = [math.hypot(bx - o["x_mm"], by - o["y_mm"]) for o in f]
                j = int(np.argmin(dists))
                if dists[j] < 500.0:
                    target = f[j]
                    break
            if target is None:
                continue

            disp = math.hypot(target["x_mm"] - bx, target["y_mm"] - by)
            if disp > best_disp:
                best_disp = disp
                best = target

        if best is None or best_disp < movement_threshold_mm:
            return None

        conf = min(1.0, (best_disp / (movement_threshold_mm * 2.0)) * (valid_windows / max(2, window_scans)))
        return Opponent(
            x_mm=float(best["x_mm"]),
            y_mm=float(best["y_mm"]),
            displacement_mm=float(best_disp),
            confidence=float(conf),
        )

    def snapshot(self, opponent: Optional[Opponent], seq: int = 0) -> Dict:
        health = "ok"
        if self.empty_scan_streak >= 5:
            health = "degraded"
        if self.empty_scan_streak >= 20:
            health = "no_scan"

        return {
            "timestamp": int(time.time()),
            "seq": seq,
            "port": self.port,
            "uptime_s": int(time.time()) - self.started_at if self.started_at else 0,
            "last_points_count": self.last_points_count,
            "empty_scan_streak": self.empty_scan_streak,
            "health": health,
            "pillars_count": len(self.pillars),
            "pillars": [asdict(p) for p in self.pillars],
            "opponent_detected": opponent is not None,
            "opponent": asdict(opponent) if opponent else None,
        }

    @staticmethod
    def write_json(path: str, data: Dict) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
