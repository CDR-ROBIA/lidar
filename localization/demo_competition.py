"""
Demo runner for competition adapter: performs initial pillar calibration
then runs opponent detection for a short period and saves results.
"""
from __future__ import annotations
import argparse
import time

from competition_adapter import CompetitionAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=str, default=None)
    parser.add_argument('--min-pillars', type=int, default=3)
    parser.add_argument('--max-pillars', type=int, default=6)
    parser.add_argument('--scans', type=int, default=40)
    parser.add_argument('--windows', type=int, default=10)
    parser.add_argument('--out', type=str, default='competition_cycle.json')
    args = parser.parse_args()

    adapter = CompetitionAdapter(port=args.port)
    try:
        adapter.start_lidar()
        print('Running full competition cycle...')
        result = adapter.run_cycle(
            min_pillars=args.min_pillars,
            max_pillars=args.max_pillars,
            calibration_scans=args.scans,
            watch_windows=args.windows,
        )

        print('Result summary:')
        print(f"- pillars_count: {result['pillars_count']}")
        print(f"- opponent_detected: {result['opponent_detected']}")
        if result['best_opponent']:
            print(f"- best_opponent: {result['best_opponent']}")

        adapter.export_calibration(args.out)
        print(f"Exported result to {args.out}")

    finally:
        adapter.stop_lidar()


if __name__ == '__main__':
    main()
