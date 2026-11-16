"""
Detección de pares de fotos superpuestas a partir de sus footprints georreferenciados.
Con fallback por distancia GPS (Haversine) si los footprints no intersectan.
"""
import math
from typing import List, Dict, Tuple
from shapely.strtree import STRtree
from shapely.geometry import Polygon


def find_overlapping_pairs(
    photos: List[dict],
    min_overlap_m2: float = 1.0,
    use_fallback: bool = True,
) -> List[dict]:
    """
    Encuentra todos los pares de fotos cuyos footprints se intersectan.

    Parámetros
    ----------
    photos : list of dict
        Cada dict debe contener al menos 'path', 'latitude', 'longitude'
        y una clave 'footprint' con el resultado de compute_footprint().
    min_overlap_m2 : float
        Área mínima de intersección (m²) para considerar que hay superposición.
    use_fallback : bool
        Si es True y no hay intersecciones válidas, busca pares cercanos por GPS.

    Retorna
    -------
    Lista de dicts, cada uno representando un par superpuesto:
        {
            'photo_a': dict,
            'photo_b': dict,
            'intersection_polygon': shapely Polygon,
            'intersection_area_m2': float,
            'overlap_ratio_a': float,  # area_interseccion / area_footprint_a
            'overlap_ratio_b': float,
            'midline': list,  # [(x1,y1), (x2,y2)] línea media entre centroides
        }
    """
    # Construir índice espacial para acelerar búsqueda
    polys = []
    valid_indices = []
    for i, p in enumerate(photos):
        fp = p.get('footprint')
        if fp and fp.get('polygon'):
            polys.append(fp['polygon'])
            valid_indices.append(i)

    if not polys:
        return []

    tree = STRtree(polys)
    # Mapeo: identidad del objeto Polygon -> índice original en photos
    poly_to_idx = {id(p): idx for p, idx in zip(polys, valid_indices)}
    pairs = []
    seen = set()

    for idx_a in valid_indices:
        photo_a = photos[idx_a]
        poly_a = photo_a['footprint']['polygon']

        # Fotos candidatas que intersectan el bounding box de poly_a
        candidates = tree.query(poly_a)
        for candidate in candidates:
            # Shapely 2 devuelve índices (numpy.integer); Shapely 1 devolvía
            # los polígonos. Admitir ambos formatos es imprescindible: tratar
            # un índice como polígono hacía descartar todas las intersecciones.
            if hasattr(candidate, 'geom_type'):
                idx_b = poly_to_idx.get(id(candidate))
            else:
                candidate_position = int(candidate)
                if candidate_position < 0 or candidate_position >= len(valid_indices):
                    continue
                idx_b = valid_indices[candidate_position]
            if idx_b is None or idx_b <= idx_a:
                continue

            pair_key = (min(idx_a, idx_b), max(idx_a, idx_b))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            photo_b = photos[idx_b]
            poly_b = photo_b['footprint']['polygon']

            if not poly_a.intersects(poly_b):
                continue

            inter = poly_a.intersection(poly_b)
            if inter.is_empty:
                continue

            area = inter.area
            if area < min_overlap_m2:
                continue

            # Línea media entre centroides (frontera geométrica por defecto)
            cx_a, cy_a = photo_a['footprint']['centroid_utm']
            cx_b, cy_b = photo_b['footprint']['centroid_utm']
            midline = [(cx_a, cy_a), (cx_b, cy_b)]

            pairs.append({
                'photo_a': photo_a,
                'photo_b': photo_b,
                'intersection_polygon': inter,
                'intersection_area_m2': area,
                'overlap_ratio_a': area / poly_a.area if poly_a.area > 0 else 0,
                'overlap_ratio_b': area / poly_b.area if poly_b.area > 0 else 0,
                'midline': midline,
            })

    # ------------------------------------------------------------------
    # FALLBACK: si no hay pares por footprint, usar distancia GPS directa
    # (Haversine). Esto ayuda cuando las coordenadas GPS son imprecisas
    # o el footprint se calcula mal.
    # ------------------------------------------------------------------
    if not pairs and use_fallback:
        pairs = _fallback_by_gps_distance(photos, valid_indices)

    return pairs


def _haversine(lat1, lon1, lat2, lon2):
    """Distancia en metros entre dos puntos (lat, lon) en grados."""
    R = 6371000  # radio de la Tierra en metros
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _fallback_by_gps_distance(photos, valid_indices):
    """
    Cuando los footprints no intersectan, usa distancia GPS (Haversine)
    para encontrar pares cercanos. El umbral es el ancho promedio del
    footprint multiplicado por 1.5.
    """
    # Calcular umbral: ancho promedio de footprint * 1.5
    widths = []
    for idx in valid_indices:
        fp = photos[idx].get('footprint')
        if fp:
            widths.append(fp.get('width_m', 0))
    if not widths:
        return []

    avg_width = sum(widths) / len(widths)
    threshold = avg_width * 1.5

    pairs = []
    seen = set()
    for i in range(len(valid_indices)):
        for j in range(i + 1, len(valid_indices)):
            idx_a = valid_indices[i]
            idx_b = valid_indices[j]
            pair_key = (idx_a, idx_b)
            if pair_key in seen:
                continue
            seen.add(pair_key)

            p_a = photos[idx_a]
            p_b = photos[idx_b]
            lat_a = p_a.get('latitude')
            lon_a = p_a.get('longitude')
            lat_b = p_b.get('latitude')
            lon_b = p_b.get('longitude')
            if None in (lat_a, lon_a, lat_b, lon_b):
                continue

            dist = _haversine(lat_a, lon_a, lat_b, lon_b)
            if dist <= threshold:
                # Frontera geométrica: línea entre centroides UTM
                fp_a = p_a.get('footprint')
                fp_b = p_b.get('footprint')
                if fp_a and fp_b:
                    cx_a, cy_a = fp_a['centroid_utm']
                    cx_b, cy_b = fp_b['centroid_utm']
                    midline = [(cx_a, cy_a), (cx_b, cy_b)]
                else:
                    midline = None

                pairs.append({
                    'photo_a': p_a,
                    'photo_b': p_b,
                    'intersection_polygon': None,
                    'intersection_area_m2': 0.0,
                    'overlap_ratio_a': 0.0,
                    'overlap_ratio_b': 0.0,
                    'midline': midline,
                    'fallback_gps_distance_m': dist,
                })

    return pairs
