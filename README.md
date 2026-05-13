# LiDAR Robot Localization

Système de localisation robotique par LiDAR, refactorisé en Python orienté objet.

Le dépôt contient uniquement le noyau utile :

- `lidar/lidar.py` : classe principale `Lidar`
- `lidar/localization.py` : fonctions pures de trilatération et d'estimation du heading
- `lidar/state.py` : dataclass d'état robot
- `lidar/visualizer.py` : visualisation matplotlib séparée
- `lidar/main.py` : exemple minimal d'utilisation
- `detection_poteaux.py` : détection des poteaux via `PoleDetector`

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation minimale

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

## API principale

- `Lidar.start()` : démarre le LiDAR et les threads internes
- `Lidar.stop()` : arrête proprement le système
- `Lidar.get_pos()` : renvoie `(x_mm, y_mm, heading_rad)` ou `None`
- `Lidar.is_ready()` : indique si une position valide a déjà été calculée
- `Lidar.get_raw_scan()` : renvoie le dernier scan brut

## Arborescence

```text
lidar-main/
├── detection_poteaux.py
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
