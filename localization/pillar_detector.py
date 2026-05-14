"""
Pillar detection utilities.

Provides functions to detect vertical cylindrical pillars from LiDAR point
clouds. Does not depend on external libraries.
"""
from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np


def cluster_points(points: np.ndarray, dist_thresh: float = 100.0) -> List[np.ndarray]:
    """Simple spatial clustering using iterative region growing.

    Args:
        points: (N,2) array
        dist_thresh: maximum Euclidean distance to consider neighbor (mm)

    Returns:
        list of clusters (each is numpy array (M,2))
    """
    if points is None or len(points) == 0:
        return []

    pts = points.copy()
    N = pts.shape[0]
    used = np.zeros(N, dtype=bool)
    clusters = []

    for i in range(N):
        if used[i]:
            continue
        seed = pts[i:i+1]
        stack = [i]
        comp_idx = []
        used[i] = True
        while stack:
            idx = stack.pop()
            comp_idx.append(idx)
            # find neighbors
            d2 = np.sum((pts - pts[idx])**2, axis=1)
            neigh = np.where((d2 <= dist_thresh**2) & (~used))[0]
            for j in neigh:
                used[j] = True
                stack.append(j)

        comp = pts[comp_idx]
        clusters.append(comp)

    return clusters


def detect_pillars(points: np.ndarray,
                   min_points: int = 8,
                   max_radius_mm: float = 400.0,
                   dist_thresh: float = 80.0) -> List[Dict]:
    """Detect candidate pillars from point cloud in robot frame.

    Heuristic: cluster spatially; for each cluster compute centroid and
    approximate radius; accept clusters with enough points and small radius.

    Returns list of dicts: {"centroid":(x,y), "radius_mm":r, "points_count":n}
    """
    if points is None or len(points) == 0:
        return []

    clusters = cluster_points(points, dist_thresh=dist_thresh)
    candidates = []
    for comp in clusters:
        if len(comp) < min_points:
            continue
        centroid = comp.mean(axis=0)
        dists = np.linalg.norm(comp - centroid, axis=1)
        radius = float(np.percentile(dists, 90))
        # reject very large clusters
        if radius > max_radius_mm:
            continue
        candidates.append({
            'centroid': (float(centroid[0]), float(centroid[1])),
            'radius_mm': radius,
            'points_count': int(len(comp))
        })

    # sort by distance (nearer first)
    candidates.sort(key=lambda c: (c['centroid'][0]**2 + c['centroid'][1]**2))
    return candidates


def filter_pillars_by_consistency(detections_history: List[List[Dict]],
                                   min_observations: int = 3,
                                   position_tol_mm: float = 200.0) -> List[Dict]:
    """From history of detections (list per scan), keep pillars observed at least
    `min_observations` times clustered by proximity.

    Returns consolidated list of pillars with averaged positions and counts.
    """
    all = []
    # flatten with timestamp index
    for scan_idx, dets in enumerate(detections_history):
        for d in dets:
            x, y = d['centroid']
            all.append({'x': x, 'y': y, 'scan': scan_idx})

    if not all:
        return []

    pts = np.array([[a['x'], a['y']] for a in all])
    used = np.zeros(len(pts), dtype=bool)
    groups = []
    for i in range(len(pts)):
        if used[i]:
            continue
        group_idx = [i]
        used[i] = True
        for j in range(i+1, len(pts)):
            if used[j]:
                continue
            if np.linalg.norm(pts[j] - pts[i]) <= position_tol_mm:
                used[j] = True
                group_idx.append(j)
        groups.append(group_idx)

    consolidated = []
    for g in groups:
        scans_seen = set()
        xs = []
        ys = []
        for idx in g:
            a = all[idx]
            xs.append(a['x'])
            ys.append(a['y'])
            scans_seen.add(a['scan'])
        if len(scans_seen) >= min_observations:
            consolidated.append({
                'centroid': (float(np.mean(xs)), float(np.mean(ys))),
                'observations': len(scans_seen),
                'points_total': len(g)
            })

    # sort by observations desc
    consolidated.sort(key=lambda c: (-c['observations'], c['centroid'][0]**2 + c['centroid'][1]**2))
    return consolidated
