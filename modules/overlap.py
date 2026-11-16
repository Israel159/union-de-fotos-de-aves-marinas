"""
Detección de pares de fotos superpuestas a partir de sus footprints georreferenciados.
"""
from typing import List, Dict, Tuple
from shapely.strtree import STRtree
from shapely.geometry import Polygon


def find_overlapping_pairs(photos: List[dict], min_overlap_m2: float = 1.0) -> List[dict]:
    """
    Encuentra todos los pares de fotos cuyos footprints se intersectan.

    Parámetros
    ----------
    photos : list of dict
        Cada dict debe contener al menos 'path', 'latitude', 'longitude'
        y una clave 'footprint' con el resultado de compute_footprint().
    min_overlap_m2 : float
        Área mínima de intersección (m²) para considerar que hay superposición.

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
        for candidate_poly in candidates:
            # Recuperar índice original usando identidad del objeto (id())
            idx_b = poly_to_idx.get(id(candidate_poly))
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

    return pairs
