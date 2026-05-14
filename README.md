# LiDAR Robot Localization

Système LiDAR pour robot avec deux couches:

- couche historique dans `lidar/` (inchangée)
- couche d'adaptation compétition dans `localization/` (calibration piliers + détection adversaire)

## Installation

```bash
pip install -r requirements.txt
```

Sous Linux, assure-toi aussi d'avoir accès au port série du LiDAR, par exemple:

```bash
sudo usermod -aG dialout $USER
```

Puis déconnecte/reconnecte ta session ou redémarre pour que le groupe soit pris en compte.

## Utilisation minimale (couche historique)

```python
from lidar import Lidar

with Lidar(port="COM5") as lidar:
    lidar.start()
    pos = lidar.get_pos()
    if pos:
        x, y, heading = pos
        print(x, y, heading)
```

## Lancement de l'exemple

```bash
python3 -m lidar.main
```

Pour lancer directement le visualiseur LiDAR:

```bash
python3 lidar/visualizer.py
```

## Utilisation compétition (simple et propre)

Pipeline recommandé sans modifier `lidar/*`:

1. calibration initiale de `3..6` piliers
2. détection d'adversaire mobile
3. sortie JSONL propre pour la stratégie

Commande:

```bash
cd /home/ben/Downloads/lidar-main
sg dialout -c '.venv/bin/python localization/run_simple_competition.py --detect-timeout 30 --min-pillars 3 --max-pillars 6 --calib-scans 40 --loops 20 --out competition_data.jsonl'
```

Documentation dédiée: `localization/SIMPLE_COMPETITION_README.md`

## API principale

- `Lidar.start()` : démarre le LiDAR et les threads internes
- `Lidar.stop()` : arrête proprement le système
- `Lidar.get_pos()` : renvoie `(x_mm, y_mm, heading_rad)` ou `None`
- `Lidar.is_ready()` : indique si une position valide a déjà été calculée
- `Lidar.get_raw_scan()` : renvoie le dernier scan brut

## Arborescence (principale)

```text
lidar-main/
├── detection_poteaux.py
├── localization/
│   ├── simple_competition.py
│   ├── run_simple_competition.py
│   └── SIMPLE_COMPETITION_README.md
├── requirements.txt
└── lidar/
    ├── __init__.py
    ├── lidar.py
    ├── localization.py
    ├── main.py
    ├── state.py
    └── visualizer.py
```

## Notes

- Le système attend un port LiDAR valide, par exemple `COM5` sous Windows ou `/dev/ttyUSB0` sous Linux.
- Les algorithmes métier sont conservés : détection via `PoleDetector`, trilatération, estimation du heading.
- La visualisation est séparée de la logique de localisation.
- Les nouveaux scripts de compétition n'éditent pas le dossier `lidar/`.
