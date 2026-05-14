#!/usr/bin/env python3
"""
Diagnostic rapide pour vérifier la présence du LiDAR sur la Raspberry Pi.

Usage:
    python3 check_lidar.py
    python3 check_lidar.py /dev/ttyUSB0
"""

from __future__ import annotations

import glob
import os
import sys
from typing import List, Optional


def list_serial_candidates() -> List[str]:
    """Return likely serial device paths for a LiDAR."""
    candidates = []
    candidates.extend(sorted(glob.glob("/dev/ttyUSB*")))
    candidates.extend(sorted(glob.glob("/dev/ttyACM*")))
    candidates.extend(sorted(glob.glob("/dev/serial/by-id/*")))
    return candidates


def try_import_rplidar() -> Optional[object]:
    """Import the RPLidar driver if available."""
    try:
        from rplidar import RPLidar  # type: ignore
        return RPLidar
    except Exception as exc:
        print(f"[ERROR] Impossible d'importer rplidar: {exc}")
        return None


def main() -> int:
    """Run the diagnostic checks."""
    port = sys.argv[1] if len(sys.argv) > 1 else None

    print("=== Diagnostic LiDAR ===")
    print(f"User: {os.getenv('USER', 'unknown')}")
    print(f"Python: {sys.version.split()[0]}")

    print("\n[1] Périphériques série détectés:")
    candidates = list_serial_candidates()
    if candidates:
        for candidate in candidates:
            print(f"  - {candidate}")
    else:
        print("  Aucun /dev/ttyUSB*, /dev/ttyACM* ou /dev/serial/by-id trouvé")

    print("\n[2] Import du driver:")
    rplidar_cls = try_import_rplidar()
    if rplidar_cls is None:
        print("  Installe le driver avec: pip install rplidar-roboticscape")
        return 1
    print("  OK")

    if port is None:
        if candidates:
            port = candidates[0]
            print(f"\n[3] Port utilisé par défaut: {port}")
        else:
            print("\n[3] Aucun port fourni et aucun périphérique détecté.")
            print("  Lance le script avec un port explicite, par exemple:")
            print("  python3 check_lidar.py /dev/ttyUSB0")
            return 1
    else:
        print(f"\n[3] Port demandé: {port}")

    print("\n[4] Test de connexion:")
    lidar = None
    try:
        lidar = rplidar_cls(port)
        lidar.stop()
        lidar.stop_motor()
        lidar.start_motor()
        print("  Connexion OK")

        print("\n[5] Test d'un scan:")
        scan_iterator = lidar.iter_scans(max_buf_meas=500)
        scan = next(scan_iterator, None)
        if scan is None:
            print("  Aucun scan reçu")
            return 2

        print(f"  Scan reçu: {len(scan)} mesures")
        print("  Aperçu des 5 premières mesures:")
        for measurement in scan[:5]:
            print(f"    {measurement}")
        return 0

    except Exception as exc:
        print(f"[ERROR] Test LiDAR échoué: {exc}")
        return 2
    finally:
        if lidar is not None:
            try:
                lidar.stop()
            except Exception:
                pass
            try:
                lidar.stop_motor()
            except Exception:
                pass
            try:
                lidar.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
