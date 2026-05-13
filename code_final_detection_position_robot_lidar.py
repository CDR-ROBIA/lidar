import numpy as np
from rplidar import RPLidar
import time
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from detection_poteaux import PoleDetector  # importation depuis le fichier de detection des poteaux

# =========================
# CONFIGURATION
# =========================
PORT = "COM5"
BAUDRATE = 115200

### taille du terrain
TERRAIN_X = 3000
TERRAIN_Y = 2000

# distance maximale acceptée = diagonale du terrain (un point plus loin ne peut pas être dans l'arène)
DIST_MAX = int(np.sqrt(TERRAIN_X**2 + TERRAIN_Y**2))

# =========================
# POSITION DES POTEAUX (mm)
# =========================
POTEAU_1 = np.array([0,    0   ])  # bas gauche
POTEAU_2 = np.array([3000, 0   ])  # bas droite
POTEAU_3 = np.array([3000, 2000])  # haut droite

# =========================
# VALEURS ROBOSTUDIO
# =========================

# poissiblement pas besoin on verra si c'est utile 

DIST_P2_P3 = 2000  # distance fixe entre P2 et P3 en mm #commentaire du commentaire : c'est la largeur du térrain vu qu'on part sur 3 poteaux sur les cotés

TOLERANCE_DIST  = 150   # tolérance distance pour chercher un poteau (mm)
TOLERANCE_ANGLE = 10    # tolérance angulaire une fois le heading connu (degrés)


# =========================
# DONNÉES PARTAGÉES ENTRE LES THREADS ET L'AFFICHAGE
# =========================
# Ce dictionnaire est le point central de communication entre :
#   - thread_lidar     → écrit "scan" et "num_scan"
#   - thread_detection → écrit la position, le heading, les distances et les angles
#   - update()         → lit tout pour l'affichage
# Toutes les écritures/lectures sont protégées par data_lock (voir ci-dessous)
data = {
    "robot_x"   : 0.0,
    "robot_y"   : 0.0,
    "heading"   : 0.0,   # orientation du robot dans le repère absolu (rad)
    "scan"      : [],
    "r1"        : None,
    "r2"        : None,
    "r3"        : None,
    "angle_p1"  : 0.0,
    "angle_p2"  : 0.0,
    "angle_p3"  : 0.0,
    "num_scan"  : 0,
    "status"    : "démarrage..."
}

# Verrou pour protéger les données partagées entre les threads
# (évite les lectures partielles pendant une écriture)
# Utilisation : "with data_lock:" avant chaque accès à data
data_lock = threading.Lock()

# =========================
# CONNEXION LIDAR
# =========================

lidar = RPLidar(PORT, baudrate=BAUDRATE)
lidar.stop()        # arrêt propre au cas où le LiDAR serait déjà en marche
lidar.stop_motor()
time.sleep(1)       # pause pour laisser le temps au moteur de s'arrêter complètement
lidar.start_motor()
time.sleep(1)       # pause pour laisser le temps au moteur d'atteindre sa vitesse nominale

# iter_scans retourne un générateur : chaque appel à next() donne un scan complet (360°)
# max_buf_meas=500 : taille du buffer interne de la lib rplidar, augmenté pour absorber
# les rafales de points sans déclencher l'avertissement "Too many measurements"
scan_iterator = lidar.iter_scans(max_buf_meas=500)

# =========================
# INSTANCIATION DU DÉTECTEUR DE POTEAUX
# =========================
# buffer_size=4 → utilise les 4 derniers scans complets pour la détection.
# Cet objet est partagé entre thread_lidar (add_scan) et thread_detection (detect)
# La classe PoleDetector gère elle-même la thread-safety de son buffer interne
detector = PoleDetector(buffer_size=4)


# =========================
# TRILATÉRATION
# =========================
# Principe : connaissant la position exacte de 3 poteaux (p1, p2, p3)
# et les distances mesurées depuis le robot (r1, r2, r3),
# on résout un système linéaire 2x2 obtenu en soustrayant l'équation
# du cercle de p1 à celles de p2 et p3, ce qui élimine les termes x² et y².
# Toutes les valeurs doivent être dans la même unité (ici : mm).
def trilateration(p1, r1, p2, r2, p3, r3):
    # Construction de la matrice A (coefficients en x et y)
    A = np.array([
        [2*(p2[0]-p1[0]), 2*(p2[1]-p1[1])],
        [2*(p3[0]-p1[0]), 2*(p3[1]-p1[1])]
    ])
    # Construction du vecteur B (termes constants)
    B = np.array([
        r1**2 - r2**2 + p2[0]**2 - p1[0]**2 + p2[1]**2 - p1[1]**2,
        r1**2 - r3**2 + p3[0]**2 - p1[0]**2 + p3[1]**2 - p1[1]**2
    ])
    # Vérification que le système est bien déterminé (det ≠ 0)
    # Si det ≈ 0, les 3 poteaux sont quasi-alignés → solution indéterminée
    det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
    if abs(det) < 1e-6:
        return None
    return np.linalg.solve(A, B)


# =========================
# ESTIMATION DU HEADING À PARTIR D'UN POTEAU
# =========================
def estimer_heading(pos, poteau, angle_lidar_mesure):
    """Estime l'orientation du robot à partir de la position connue d'un poteau,
    la position estimée du robot, et l'angle LIDAR mesuré vers ce poteau.
    """
    # angle absolu (trigo) du poteau vu depuis le robot
    angle_absolu = np.arctan2(poteau[1] - pos[1], poteau[0] - pos[0])
    # angle LIDAR = -(angle_absolu - heading)  =>  heading = angle_absolu + angle_lidar_rad
    angle_lidar_rad = np.radians(angle_lidar_mesure)
    heading = angle_absolu + angle_lidar_rad
    return heading


# =========================
# THREAD 1 — LECTURE LIDAR (ultra-léger, ne fait qu'alimenter le buffer)
# Ce thread tourne aussi vite que possible pour ne jamais saturer le buffer série.
# Il ne fait aucun calcul lourd : juste lire + stocker le scan brut.
# =========================
def thread_lidar():
    while True:
        # next() bloque jusqu'à ce qu'un scan complet soit disponible
        scan = next(scan_iterator, None)
        if scan is None:
            with data_lock:
                data["status"] = "Erreur lecture scan"
            continue

        # Mise à jour du scan brut et du compteur (sous verrou car lu par l'affichage)
        with data_lock:
            data["num_scan"] += 1
            data["scan"] = scan   # scan brut pour l'affichage

        # Alimentation du buffer du détecteur — opération O(n) très rapide
        detector.add_scan(scan)


# =========================
# THREAD 2 — DÉTECTION ET CALCUL DE POSITION (calculs lourds isolés)
# Ce thread tourne à sa propre cadence et ne bloque jamais la lecture LiDAR.
# Il appelle DBSCAN puis trilatération dès qu'un nouveau scan est disponible.
# =========================
def thread_detection():
    heading = 0.0  # orientation du robot (rad), sera estimée

    while True:
        # Petite pause pour ne pas saturer le CPU entre deux détections
        # Le LiDAR tourne à ~9 Hz → un scan toutes les ~110 ms
        # On détecte à ~15 Hz (toutes les 60 ms) → on ne rate aucun scan
        time.sleep(0.06)

        # On attend qu'au moins un scan soit dans le buffer avant de détecter
        if not detector.ready:
            continue

        # Détection des 3 poteaux sur les 4 derniers scans fusionnés
        # Retourne : {"P1": {"angle": °, "distance": mm}, ...} ou None
        poles = detector.detect()
        if poles is None:
            with data_lock:
                data["status"] = "Poteau manquant"
            continue

        # Extraction des distances (mm) et angles (°) détectés pour chaque poteau
        r1, a1 = poles["P1"]["distance"], poles["P1"]["angle"]
        r2, a2 = poles["P2"]["distance"], poles["P2"]["angle"]
        r3, a3 = poles["P3"]["distance"], poles["P3"]["angle"]

        
        # r1/r2/r3 sont en mm, POTEAU_x en mm → tout cohérent
        pos = trilateration(POTEAU_2, r2, POTEAU_3, r3, POTEAU_1, r1)
        if pos is None:
            with data_lock:
                data["status"] = "trilatération impossible"
            continue

        if pos[0]> 3000 or pos[1] > 2000 or pos[0]<0 or pos[1] < 0: #si la position calculée est en dehors des limites alors on l'ignore
            continue

        # Estimer le heading à partir des 3 poteaux (moyenne circulaire)
        h1 = estimer_heading(pos, POTEAU_1, a1)
        h2 = estimer_heading(pos, POTEAU_2, a2)
        h3 = estimer_heading(pos, POTEAU_3, a3)
        # Moyenne circulaire via sin/cos
        # (on ne peut pas faire une moyenne directe sur des angles car 359° et 1° donneraient 180°)
        heading = np.arctan2(
            (np.sin(h1) + np.sin(h2) + np.sin(h3)) / 3,
            (np.cos(h1) + np.cos(h2) + np.cos(h3)) / 3
        )

        # Écriture des résultats dans le dict partagé (protégée par le verrou)
        with data_lock:
            data["robot_x"]  = pos[0]
            data["robot_y"]  = pos[1]
            data["heading"]  = heading
            data["r1"]       = r1
            data["r2"]       = r2
            data["r3"]       = r3
            data["angle_p1"] = a1
            data["angle_p2"] = a2
            data["angle_p3"] = a3
            data["status"]   = "OK"

        print(f"[Scan #{data['num_scan']}]  X={int(pos[0])}mm  Y={int(pos[1])}mm")


# Démarrage des deux threads en parallèle
# daemon=True : les threads s'arrêtent automatiquement quand le programme principal se termine
t_lidar     = threading.Thread(target=thread_lidar,     daemon=True)
t_detection = threading.Thread(target=thread_detection, daemon=True)
t_lidar.start()
t_detection.start()


# =========================
# AFFICHAGE MATPLOTLIB
# =========================
fig, ax = plt.subplots(figsize=(15, 14))
ax.set_facecolor("#0d0d0d")
fig.patch.set_facecolor("#0d0d0d")

def update(frame):
    # Lecture snapshot des données partagées (protégée par le verrou)
    # On copie tout d'un coup pour minimiser le temps de verrouillage
    with data_lock:
        rx       = data["robot_x"]
        ry       = data["robot_y"]
        heading  = data["heading"]
        scan     = list(data["scan"])   # list() pour copier le scan et libérer le verrou rapidement
        r1       = data["r1"]
        r2       = data["r2"]
        r3       = data["r3"]
        a1       = data["angle_p1"]
        a2       = data["angle_p2"]
        a3       = data["angle_p3"]
        num_scan = data["num_scan"]
        status   = data["status"]

    # Effacement et reconfiguration du graphe à chaque frame
    ax.cla()
    ax.set_facecolor("#0d0d0d")
    ax.set_xlim(-300, 3400)
    ax.set_ylim(-300, 2400)
    ax.set_aspect("equal")
    ax.grid(True, color="#333333", linewidth=0.5)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    # --- points du scan brut (convertis en XY depuis le robot) ---
    # On n'affiche qu'1 point sur 3 pour réduire la charge graphique
    if len(scan) > 0:
        xs, ys = [], []
        for i, mesure in enumerate(scan):
            if i % 3 != 0:   # sous-échantillonnage : 1 point sur 3 affiché
                continue
            angle_lidar_deg = mesure[1]  # angle LIDAR (sens horaire)
            d = mesure[2]
            if d <= 0:
                continue
            if d > DIST_MAX:  # si distance du point dépasse l'arene il n'est pas pris en compte
                continue
            # Conversion angle LIDAR → angle absolu (trigo)
            # LIDAR horaire : angle_absolu = heading - angle_lidar
            angle_absolu = heading - np.radians(angle_lidar_deg)
            # position absolue du point détecté
            px = rx + d * np.cos(angle_absolu)
            py = ry + d * np.sin(angle_absolu)
            xs.append(px)
            ys.append(py)
        ax.scatter(xs, ys, s=2, color="#00aaff", alpha=0.4, label="scan")

    # --- poteaux fixes ---
    for poteau, nom in [(POTEAU_1, "P1"), (POTEAU_2, "P2"), (POTEAU_3, "P3")]:
        ax.plot(poteau[0], poteau[1], "o", color="#ff4444", markersize=12)
        ax.annotate(nom, poteau, textcoords="offset points", xytext=(8, 8),
                    color="#ff4444", fontsize=11, fontweight="bold")

    # --- robot ---
    ax.plot(rx, ry, "o", color="#00ff88", markersize=14, zorder=5)
    ax.annotate("ROBOT", (rx, ry), textcoords="offset points", xytext=(10, 10),
                color="#00ff88", fontsize=11, fontweight="bold")

    # --- lignes robot → poteaux ---
    for poteau, r, nom in [
        (POTEAU_1, r1, "P1"),
        (POTEAU_2, r2, "P2"),
        (POTEAU_3, r3, "P3")
    ]:
        if r is not None:
            ax.plot([rx, poteau[0]], [ry, poteau[1]], "--", color="#ffaa00", alpha=0.5, linewidth=1)
            mx, my = (rx + poteau[0]) / 2, (ry + poteau[1]) / 2
            ax.annotate(f"{int(r)}mm", (mx, my), color="#ffaa00", fontsize=9)

    # --- infos texte ---
    status_color = "#00ff88" if status == "OK" else "#ff4444"
    titre = (
        f"Scan #{num_scan}   |   "
        f"X={int(rx)}mm  Y={int(ry)}mm   |   "
        f"Status: {status}"
    )
    ax.set_title(titre, color=status_color, fontsize=11, pad=10)

    # Boîte d'informations en haut à gauche : angles et distances vers chaque poteau
    info = (
        f"P1 → angle={a1:.0f}°  dist={int(r1) if r1 else '?'}mm\n"
        f"P2 → angle={a2:.0f}°  dist={int(r2) if r2 else '?'}mm\n"
        f"P3 → angle={a3:.0f}°  dist={int(r3) if r3 else '?'}mm\n"
        f"Heading: {np.degrees(heading):.1f}°"
    )
    ax.text(0.02, 0.98, info, transform=ax.transAxes, color="white",
            fontsize=9, verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="#1a1a1a", alpha=0.8))

    #tracer la zone d'evolution du robot
    P1 = np.array([0,    0   ])  # bas gauche
    P2 = np.array([3000, 0   ])  # bas droite
    P3 = np.array([3000, 2000])  # haut droite
    P4 = np.array([0,    2000])  # haut gauche
    # Tracer le carré
    carre = plt.Polygon([P1, P2, P3, P4], linewidth=2, edgecolor='cyan', facecolor='none')
    ax.add_patch(carre)


# interval=300 ms → ~3 fps pour l'affichage, largement suffisant visuellement
# et beaucoup moins gourmand que 200 ms
# cache_frame_data=False : évite l'accumulation de frames en mémoire (warning matplotlib)
ani = animation.FuncAnimation(fig, update, interval=300, cache_frame_data=False)

try:
    plt.show()
except KeyboardInterrupt:
    print("Arrêt")
finally:
    # Arrêt propre du LiDAR dans tous les cas (même si exception)
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()