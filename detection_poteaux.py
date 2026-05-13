"""
Détection temps-réel des 3 poteaux — RP LiDAR A2M8
====================================================
Carte 3 m × 2 m :
  P1 (0,0) bas-gauche    P1↔P2 = 3.0 m
  P2 (3,0) bas-droite    P2↔P3 = 2.0 m
  P3 (3,2) haut-droite   P1↔P3 = √13 ≈ 3.606 m

Architecture temps-réel :
  • PoleDetector  : objet permanent, garde les N derniers scans en mémoire (deque)
  • add_scan()    : appelé à chaque nouveau scan complet du LiDAR
  • detect()      : lance la détection sur le buffer courant, retourne P1/P2/P3

Dépendances : pip install numpy scikit-learn
"""

import numpy as np
from collections import deque
from itertools import combinations
from sklearn.cluster import DBSCAN

# >>> Importations :
# >>> - numpy : calculs vectoriels (coordonnées, distances, angles)
# >>> - deque : buffer circulaire pour les derniers scans
# >>> - combinations : génère tous les triplets de candidats
# >>> - DBSCAN : algorithme de clustering par densité

# ─────────────────────────────────────────────────────────────────────────────
# Distances réelles entre poteaux en MÈTRES (usage interne DBSCAN uniquement)
# La sortie de detect() est en MILLIMÈTRES pour la trilatération
# ─────────────────────────────────────────────────────────────────────────────
KNOWN_DISTANCES = {
    "P1P2": 3.0,
    "P2P3": 2.0,
    "P1P3": np.sqrt(13),   # ≈ 3.606 m  (diagonale P1→P3)
}
# >>> KNOWN_DISTANCES sert à évaluer la qualité d’un triplet de candidats.
# >>> Les distances sont en mètres car le clustering travaille en mètres.


class PoleDetector:
    """
    Maintient un buffer glissant des N derniers scans LiDAR
    et détecte les 3 poteaux à la demande.

    add_scan(scan) : appelé à chaque scan  (format rplidar natif)
    detect()       : retourne P1/P2/P3 avec angles (°) et distances en mm,
                     ou None si la détection échoue.
    """
    # >>> La classe stocke les scans bruts, agrège les points, applique DBSCAN,
    # >>> filtre les clusters, sélectionne le meilleur triplet et attribue P1,P2,P3.

    def __init__(self,
                 buffer_size=4,          # nombre de scans conservés dans le buffer
                 min_angle_coverage=300, # degrés minimum pour valider un scan complet
                 quality_min=1,          # qualité LiDAR minimum acceptée
                 dist_min_m=0.05,        # distance minimum retenue (m)
                 dist_max_m=5.5,         # distance maximum retenue (m) → hors terrain ignoré
                 eps=0.06,               # rayon de voisinage DBSCAN en mètres (6 cm)
                 min_samples=3,          # points minimum pour former un cluster
                 max_pole_radius=0.12,   # rayon max d'un poteau en m (12 cm) → filtre les murs
                 tolerance=0.50,         # tolérance en m pour l'attribution P1/P2/P3
                 known=None):

        self.buffer_size      = buffer_size
        self.min_angle_cov    = min_angle_coverage
        self.quality_min      = quality_min
        self.dist_min         = dist_min_m
        self.dist_max         = dist_max_m
        self.eps              = eps
        self.min_samples      = min_samples
        self.max_pole_radius  = max_pole_radius
        self.tolerance        = tolerance
        self.known            = known or KNOWN_DISTANCES

        # deque(maxlen=N) : expulse automatiquement le scan le plus ancien quand plein
        self._buffer = deque(maxlen=buffer_size)
        # >>> _buffer : liste circulaire des scans bruts (chacun est une liste de tuples)

    # ── API publique ──────────────────────────────────────────────────────────

    def add_scan(self, scan):
        """
        Ajoute un scan au buffer glissant.
        scan : iterable de tuples (quality, angle_deg, distance_mm)  ← format rplidar natif
        Les scans partiels (couverture angulaire < min_angle_coverage) sont ignorés.
        """
        # >>> Conversion en liste pour pouvoir extraire les angles facilement
        pts = [(q, a, d) for q, a, d in scan]
        if not pts:
            return

        # Vérifier la couverture angulaire avant d'accepter le scan
        angles = [p[1] for p in pts]
        if max(angles) - min(angles) < self.min_angle_cov:
            return   # scan partiel (début/fin de rotation), on ignore
        # >>> On refuse les scans qui ne couvrent pas assez de degrés
        # >>> (évite les scans incomplets au démarrage ou après un accrochage)

        self._buffer.append(pts)

    def detect(self):
        """
        Lance la détection sur les scans actuellement dans le buffer.

        Retourne un dict ou None :
        {
          "P1": {"angle": float (°), "distance": float (mm)},
          "P2": {...},
          "P3": {...},
          "_score"   : float,  # erreur géométrique en m², plus bas = meilleur
          "_n_scans" : int,    # nombre de scans utilisés
        }
        !! Les distances sont en MILLIMÈTRES !!
        Directement utilisables pour la trilatération (qui travaille en mm).
        Retourne None si le buffer est vide ou si moins de 3 poteaux sont trouvés.
        """
        if len(self._buffer) == 0:
            return None

        pts = self._build_point_cloud()
        if pts is None or len(pts) == 0:
            return None

        candidates = self._cluster(pts)
        if len(candidates) < 3:
            return None

        best_triplet, score = self._best_triplet(candidates)
        result = self._assign_labels(best_triplet)

        if result is None:
            return None

        result["_score"]   = round(score, 6)
        result["_n_scans"] = len(self._buffer)
        return result

    @property
    def ready(self):
        """True dès que le buffer contient au moins 1 scan valide."""
        return len(self._buffer) > 0

    # ── Méthodes internes ────────────────────────────────────────────────────

    def _build_point_cloud(self):
        """
        Agrège tous les scans du buffer en un tableau numpy (X, Y) en MÈTRES.
        Conversion : distance_mm / 1000 → distance_m
        Filtre les points hors plage ou de mauvaise qualité.
        """
        # >>> Parcourt tous les scans et tous les points, convertit les coordonnées
        # >>> polaires (angle, distance) en cartésiennes (x, y) en mètres.
        all_pts = []
        for scan in self._buffer:
            for quality, angle_deg, dist_mm in scan:
                if quality < self.quality_min:
                    continue
                dist_m = dist_mm / 1000.0   # mm → m pour le clustering interne
                if not (self.dist_min < dist_m < self.dist_max):
                    continue
                a = np.radians(angle_deg)
                all_pts.append((dist_m * np.cos(a), dist_m * np.sin(a)))
        # >>> Retourne un tableau numpy (N,2) ou None
        return np.array(all_pts) if all_pts else None

    def _cluster(self, pts):
        """
        DBSCAN sur le nuage de points (en mètres).
        Ne conserve que les clusters compacts (petits) → candidats poteaux.
        Les murs et coins forment de grands clusters → filtrés par max_pole_radius.
        """
        # >>> DBSCAN regroupe les points proches (distance < eps)
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(pts).labels_

        candidates = []
        for lbl in set(labels):
            if lbl == -1:
                continue   # bruit ignoré

            mask        = labels == lbl
            center      = pts[mask].mean(axis=0)          # centre du cluster
            radius      = np.linalg.norm(pts[mask] - center, axis=1).max()  # distance max au centre

            if radius > self.max_pole_radius:
                continue   # trop grand → mur, coin, obstacle

            # >>> Conversion du centre en coordonnées polaires (angle, distance)
            angle_deg   = float(np.degrees(np.arctan2(center[1], center[0])) % 360)
            distance_m  = float(np.linalg.norm(center))
            distance_mm = distance_m * 1000.0   # ← conversion m → mm pour la sortie finale

            candidates.append({
                "centroid"   : center,
                "angle"      : round(angle_deg,   2),
                "distance_m" : round(distance_m,  4),   # mètres  (usage interne uniquement)
                "distance_mm": round(distance_mm, 1),   # mm      (sortie vers trilatération)
            })
        # >>> candidates : liste de dictionnaires, chacun décrivant un poteau potentiel
        return candidates

    def _best_triplet(self, candidates):
        """
        Parmi toutes les combinaisons de 3 candidats, retourne celui dont les
        distances inter-centres (en mètres) correspondent le mieux aux distances réelles.
        Score = Σ (distance_mesurée - distance_attendue)²
        """
        expected = sorted(self.known.values())  # [2.0, 3.0, ~3.606]
        best_score, best_triplet = float("inf"), None

        for triplet in combinations(candidates, 3):
            c = [t["centroid"] for t in triplet]
            # >>> Calcul des trois distances mutuelles entre les centroïdes
            measured = sorted([
                np.linalg.norm(c[0] - c[1]),
                np.linalg.norm(c[1] - c[2]),
                np.linalg.norm(c[0] - c[2]),
            ])
            # >>> Somme des carrés des écarts (moindres carrés)
            score = sum((m - e) ** 2 for m, e in zip(measured, expected))
            if score < best_score:
                best_score, best_triplet = score, triplet

        return best_triplet, best_score

    def _assign_labels(self, triplet):
        """
        Identifie quel candidat est P1, P2 ou P3.
        P2 = coin à angle droit : ses deux côtés adjacents font ≈ 2 m (P2↔P3) et ≈ 3 m (P1↔P2).
        Une fois P2 identifié : P1 est à 3 m de P2, P3 à 2 m de P2.
        Retourne un dict avec distances en mm prêtes pour la trilatération.
        """
        tol = self.tolerance
        kn  = self.known

        # >>> On teste chaque candidat comme étant potentiellement P2
        for i in range(3):
            j, k = [x for x in range(3) if x != i]
            dij = np.linalg.norm(triplet[i]["centroid"] - triplet[j]["centroid"])
            dik = np.linalg.norm(triplet[i]["centroid"] - triplet[k]["centroid"])
            sides = sorted([dij, dik])

            # P2 est le seul sommet avec côtés adjacents ≈ 2 m et ≈ 3 m
            if abs(sides[0] - kn["P2P3"]) < tol and abs(sides[1] - kn["P1P2"]) < tol:
                p2 = i
                # P1 est à ~3 m de P2, P3 à ~2 m de P2
                # >>> On regarde quelle distance correspond à 3 m (P1-P2) et laquelle à 2 m (P2-P3)
                p1, p3 = (j, k) if abs(dij - kn["P1P2"]) < abs(dij - kn["P2P3"]) else (k, j)

                return {
                    name: {
                        "angle"   : triplet[idx]["angle"],
                        "distance": triplet[idx]["distance_mm"],  # mm ← pour trilatération
                    }
                    for name, idx in [("P1", p1), ("P2", p2), ("P3", p3)]
                }

        # Fallback si la tolérance est trop serrée : attribution dans l'ordre du triplet
        # >>> En cas d’échec de l’identification, on rend simplement les trois premiers
        return {
            name: {
                "angle"   : triplet[idx]["angle"],
                "distance": triplet[idx]["distance_mm"],  # mm ← pour trilatération
            }
            for name, idx in [("P1", 0), ("P2", 1), ("P3", 2)]
        }