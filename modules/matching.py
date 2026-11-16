"""Emparejamiento de detecciones y clustering de individuos."""
import math
import itertools

def _dist_m(a, b):
    """Distancia en metros entre dos puntos lat/lon (aprox. plana local)."""
    dlat = (a["lat"] - b["lat"]) * 111320.0
    dlon = (a["lon"] - b["lon"]) * 111320.0 * math.cos(math.radians((a["lat"] + b["lat"]) / 2))
    return math.hypot(dlat, dlon)

def build_candidates(detections, max_dist_m=None, dist_factor=1.2):
    """Parejas candidatas a duplicado: misma clase, fotos distintas y centros
    GPS dentro del umbral. Si max_dist_m=None, el umbral es proporcional al
    tamaño del ave en el suelo (dist_factor x su ancho), con mínimo de 1 m."""
    cands, cid = [], 0
    for i, j in itertools.combinations(range(len(detections)), 2):
        di, dj = detections[i], detections[j]
        if di["class"] != dj["class"] or di["img"] == dj["img"]:
            continue
        d = _dist_m(di, dj)
        if max_dist_m and max_dist_m > 0:
            umb = max_dist_m
        else:
            umb = max(di["size_m"], dj["size_m"]) * dist_factor
        umb = max(umb, 1.0)
        if d <= umb:
            cands.append({"id": f"c{cid}", "a": i, "b": j,
                          "dist_m": round(d, 2), "umbral_m": round(umb, 2)})
            cid += 1
    return cands

def cluster_detections(detections, candidates, decisions):
    """Union-find: une las parejas aprobadas (decision True) y devuelve
    grupos de índices de detecciones. Cada grupo = un individuo."""
    parent = list(range(len(detections)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for c in candidates:
        if decisions.get(c["id"]) is True:
            union(c["a"], c["b"])

    groups = {}
    for i in range(len(detections)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())
