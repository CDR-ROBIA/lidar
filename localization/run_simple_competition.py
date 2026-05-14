"""
Minimal CLI runner for `simple_competition.py`.

Sequence:
1) Start LiDAR
2) Calibrate 3..6 pillars
3) Repeatedly detect opponent and print/store clean JSON snapshots
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
import os

from simple_competition import SimpleCompetitionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple competition sensing runner (pillars + opponent)")
    parser.add_argument("--port", type=str, default=None)
    parser.add_argument("--no-auto-detect", action="store_true", help="Disable automatic LiDAR port detection")
    parser.add_argument("--detect-timeout", type=float, default=20.0, help="Seconds to wait for LiDAR port auto-detection")
    parser.add_argument("--min-pillars", type=int, default=3)
    parser.add_argument("--max-pillars", type=int, default=6)
    parser.add_argument("--calib-scans", type=int, default=40)
    parser.add_argument("--loops", type=int, default=20, help="Number of snapshots. Use 0 for continuous mode until Ctrl+C")
    parser.add_argument("--append", action="store_true", help="Append to output file instead of overwrite")
    parser.add_argument("--out", type=str, default="competition_data.jsonl")
    parser.add_argument("--status-out", type=str, default="competition_status.json", help="Latest snapshot JSON (atomically updated)")
    args = parser.parse_args()

    if args.min_pillars < 1 or args.max_pillars < 1 or args.min_pillars > args.max_pillars:
        raise ValueError("Invalid pillars range: expected 1 <= min-pillars <= max-pillars")
    if args.calib_scans < 5:
        raise ValueError("calib-scans must be >= 5")
    if args.loops < 0:
        raise ValueError("loops must be >= 0")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.status_out) or ".", exist_ok=True)

    run_id = str(uuid.uuid4())
    pipe = SimpleCompetitionPipeline(port=args.port)
    try:
        pipe.start(auto_detect=not args.no_auto_detect, detect_timeout_s=args.detect_timeout)
        print(f"LiDAR port in use: {pipe.port}")
        print(f"run_id={run_id} out={args.out} status={args.status_out}")
        pillars = pipe.calibrate_pillars(
            min_pillars=args.min_pillars,
            max_pillars=args.max_pillars,
            scans=args.calib_scans,
        )
        print(f"Calibration OK: {len(pillars)} pillars")

        mode = "a" if args.append else "w"
        with open(args.out, mode) as f:
            i = 0
            while True:
                if args.loops > 0 and i >= args.loops:
                    break
                opp = pipe.detect_opponent(window_scans=8, movement_threshold_mm=150.0)
                snap = pipe.snapshot(opp, seq=i)
                snap["run_id"] = run_id
                line = json.dumps(snap)
                print(line)
                f.write(line + "\n")
                f.flush()
                pipe.write_json(args.status_out, snap)
                time.sleep(0.05)
                i += 1

        print(f"Saved snapshots to {args.out}")

    except KeyboardInterrupt:
        print("Interrupted by user, clean stop.")

    finally:
        pipe.stop()


if __name__ == "__main__":
    main()
