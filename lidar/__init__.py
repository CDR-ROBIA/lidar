"""
LiDAR robot localization package.

Public API exports only:
- Lidar: main class
"""

# Lazy import to avoid requiring rplidar for basic module usage
def __getattr__(name):
    if name == "Lidar":
        from .lidar import Lidar
        return Lidar
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Lidar"]
__version__ = "2.0.0"
