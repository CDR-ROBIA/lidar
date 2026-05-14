# Competition Adaptation (No changes to existing `lidar/*`)

This adaptation layer is designed for robotics competition startup:
1. Initial calibration by detecting **3 to 6 pillars**.
2. Opponent detection after calibration.
3. Clean JSON output for downstream strategy code.

## Files added
- `localization/pillar_detector.py`
- `localization/competition_adapter.py`
- `localization/demo_competition.py`

## What it does
- Reads scans from the existing `lidar.Lidar` class (unchanged).
- Converts scans into point clouds.
- Detects stable pillar candidates across multiple scans.
- Validates pillar count is within expected range (3..6 by default).
- Detects moving candidates and excludes points near confirmed pillars.
- Exports full cycle result to JSON.

## Run
```bash
cd /home/ben/Downloads/lidar-main
sg dialout -c '.venv/bin/python localization/demo_competition.py --port /dev/ttyUSB0 --min-pillars 3 --max-pillars 6 --scans 40 --windows 10 --out competition_cycle.json'
```

## Output format (`competition_cycle.json`)
- `timestamp`: unix timestamp
- `pillars_count`: detected stable pillars count
- `pillars`: list of pillars
  - `centroid`: `[x_mm, y_mm]`
  - `observations`: scans where pillar was seen
  - `points_total`: aggregated points
- `opponent_detected`: bool
- `best_opponent`: best moving candidate or null
  - `start`: `[x_mm, y_mm]`
  - `end`: `[x_mm, y_mm]`
  - `displacement_mm`: movement over watch window
- `opponent_candidates`: list of moving candidates

## Notes
- No movement control is implemented here.
- This layer is for sensing and clean data extraction only.
- Tune thresholds inside `competition_adapter.py` and `pillar_detector.py` for your field.
