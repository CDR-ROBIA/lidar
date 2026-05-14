# Simple Competition Adaptation

This is a clean adaptation layer for robotics competition startup without modifying existing `lidar/*` files.

## What it does
- Step 1: detects and validates **3 to 6 pillars** during calibration.
- Step 2: detects a moving opponent while ignoring fixed calibrated pillars.
- Step 3: outputs clean structured data in JSONL (one snapshot per line).

## Files
- `localization/simple_competition.py`
- `localization/run_simple_competition.py`

## Run
```bash
cd /home/ben/Downloads/lidar-main
sg dialout -c '.venv/bin/python localization/run_simple_competition.py --min-pillars 3 --max-pillars 6 --calib-scans 40 --loops 20 --out competition_data.jsonl'
```

Recommended clean run (history + latest status file):
```bash
cd /home/ben/Downloads/lidar-main
sg dialout -c '.venv/bin/python localization/run_simple_competition.py --detect-timeout 30 --min-pillars 3 --max-pillars 6 --calib-scans 40 --loops 0 --out competition_data.jsonl --status-out competition_status.json'
```
Continuous mode (runs until Ctrl+C):
```bash
cd /home/ben/Downloads/lidar-main
sg dialout -c '.venv/bin/python localization/run_simple_competition.py --detect-timeout 30 --min-pillars 3 --max-pillars 6 --calib-scans 40 --loops 0 --out competition_data.jsonl'
```

Append mode (keep previous records):
```bash
cd /home/ben/Downloads/lidar-main
sg dialout -c '.venv/bin/python localization/run_simple_competition.py --append --loops 0 --out competition_data.jsonl'
```

The runner auto-detects the LiDAR serial port on Raspberry/Linux (`/dev/serial/by-id`, `/dev/ttyUSB*`, `/dev/ttyACM*`).

Optional explicit port:
```bash
sg dialout -c '.venv/bin/python localization/run_simple_competition.py --port /dev/ttyUSB0 --min-pillars 3 --max-pillars 6 --calib-scans 40 --loops 20 --out competition_data.jsonl'
```

Optional detection timeout:
```bash
sg dialout -c '.venv/bin/python localization/run_simple_competition.py --detect-timeout 30 --min-pillars 3 --max-pillars 6 --calib-scans 40 --loops 20 --out competition_data.jsonl'
```

If you do not know the port, check:
```bash
ls -l /dev/ttyUSB* /dev/ttyACM* /dev/serial/by-id/* 2>/dev/null
```

## Output snapshot fields
- `timestamp`
- `run_id`
- `seq`
- `port`
- `uptime_s`
- `last_points_count`
- `empty_scan_streak`
- `health` (`ok`, `degraded`, `no_scan`)
- `pillars_count`
- `pillars`: list of calibrated pillars (`x_mm`, `y_mm`, `radius_mm`, `observations`)
- `opponent_detected`
- `opponent` (`x_mm`, `y_mm`, `displacement_mm`, `confidence`) or `null`

## Notes
- No movement control is included.
- This layer is sensing/data only.
- Thresholds can be tuned in `simple_competition.py`.
