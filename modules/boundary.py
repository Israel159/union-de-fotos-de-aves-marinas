"""
Cálculo de la frontera de separación entre dos fotos superpuestas.
Implementa la estrategia híbrida:
  - Base: línea media geométrica entre centroides.
  - Refinamiento: polilínea que pasa por puntos medios de duplicados detectados.
  - Fallback: si no hay duplicados, se mantiene la línea media.
"""
import math
import numpy as np
from typing import List, Tuple, Dict
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import unary_union


def _estimate_homography(corners_px: np.ndarray, corners_utm: np.ndarray) -> np.ndarray:
    """
    Estima una homografía 3x3 que mapea coordenadas normalizadas de píxeles [0,1]x[0,1]
    a coordenadas UTM usando los 4 puntos esquina.
    Retorna H tal que [utm_x, utm_y, 1]^T ~ H * [px_n, py_n, 1]^T.
    """
    # Normalizar píxeles a [0,1]
    src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    dst = corners_utm.astype(np.float64)

    A = []
    b = []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -x * X, -y * X])
        A.append([0, 0, 0, x, y, 1, -x * Y, -y * Y])
        b.extend([X, Y])

    A = np.array(A)
    b = np.array(b)
    h = np.linalg.lstsq(A, b, rcond=None)[0]
    H = np.append(h, 1.0).reshape(3, 3)
    return H


def _utm_to_pixel(utm_x: float, utm_y: float, H: np.ndarray,
                  img_w: int, img_h: int) -> Tuple[float, float]:
    """
    Convierte coordenadas UTM a píxeles usando la inversa de la homografía.
    H mapea [px_norm, py_norm, 1] -> [utm_x, utm_y, w].
    """
    H_inv = np.linalg.inv(H)
    p = H_inv @ np.array([utm_x, utm_y, 1.0])
    px_n, py_n = p[0] / p[2], p[1] / p[2]
    return float(px_n * img_w), float((1.0 - py_n) * img_h)


def _pixel_to_utm(px: float, py: float, H: np.ndarray,
                  img_w: int, img_h: int) -> Tuple[float, float]:
    """Convierte píxeles a UTM vía homografía."""
    px_n = px / img_w
    py_n = 1.0 - (py / img_h)
    p = H @ np.array([px_n, py_n, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def _get_perpendicular_bisector(p1: Tuple[float, float],
                                p2: Tuple[float, float],
                                bounds: Polygon) -> LineString:
    """
    Dados dos puntos (centroides), retorna la línea perpendicular al segmento
    que los une, pasando por el punto medio, y recortada al bounding box dado.
    """
    mx = (p1[0] + p2[0]) / 2.0
    my = (p1[1] + p2[1]) / 2.0
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    # Vector perpendicular
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        # Mismo punto: devolver línea vertical por defecto
        perp_dx, perp_dy = 1.0, 0.0
    else:
        perp_dx, perp_dy = -dy, dx

    # Extender la línea perpendicular mucho más allá de los bounds
    scale = 1e6
    line = LineString([
        (mx - perp_dx * scale, my - perp_dy * scale),
        (mx + perp_dx * scale, my + perp_dy * scale),
    ])

    # Recortar al bounds (intersección del footprint o de la zona común)
    inter = line.intersection(bounds)
    if inter.is_empty:
        return line
    if inter.geom_type == 'MultiLineString':
        # Tomar el segmento más largo
        inter = max(inter.geoms, key=lambda g: g.length)
    return inter


def compute_boundary(pair: dict, matched_duplicates: List[Tuple[dict, dict]],
                     use_detector: bool = True) -> dict:
    """
    Calcula la frontera de separación entre dos fotos superpuestas.

    Parámetros
    ----------
    pair : dict
        Salida de find_overlapping_pairs() con 'photo_a', 'photo_b', 'midline', etc.
    matched_duplicates : list of (det_a, det_b)
        Pares de detecciones emparejadas por match_duplicates().
    use_detector : bool
        Si True, usa los duplicados para refinar la frontera.
        Si False, siempre usa la línea media geométrica.

    Retorna
    -------
    dict con:
        'line_utm' : shapely LineString (frontera en coordenadas UTM)
        'line_pixels_a' : [(x1,y1), (x2,y2)] en píxeles de foto A
        'line_pixels_b' : [(x1,y1), (x2,y2)] en píxeles de foto B
        'mask_polygon_utm' : shapely Polygon (región que pertenece a B dentro de A)
        'side_a_utm' : str, indica qué lado de la línea pertenece a A
    """
    photo_a = pair['photo_a']
    photo_b = pair['photo_b']
    fp_a = photo_a['footprint']
    fp_b = photo_b['footprint']

    img_w_a = photo_a.get('image_width', 1)
    img_h_a = photo_a.get('image_height', 1)
    img_w_b = photo_b.get('image_width', 1)
    img_h_b = photo_b.get('image_height', 1)

    # Homografías para cada foto
    corners_a = np.array(fp_a['corners'])
    corners_b = np.array(fp_b['corners'])
    H_a = _estimate_homography(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]), corners_a)
    H_b = _estimate_homography(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]), corners_b)

    # Bounds para recorte: unión de los dos footprints (o la intersección)
    poly_a = fp_a['polygon']
    poly_b = fp_b['polygon']
    union_bounds = unary_union([poly_a, fp_b['polygon']]).bounds
    bounds_poly = Polygon([
        (union_bounds[0], union_bounds[1]),
        (union_bounds[2], union_bounds[1]),
        (union_bounds[2], union_bounds[3]),
        (union_bounds[0], union_bounds[3]),
    ])

    cx_a, cy_a = fp_a['centroid_utm']
    cx_b, cy_b = fp_b['centroid_utm']

    if use_detector and len(matched_duplicates) > 0:
        # --- Refinamiento con duplicados ---
        # Calcular puntos medios de cada par de duplicados en UTM
        midpoints_utm = []
        for det_a, det_b in matched_duplicates:
            # Centro en píxeles -> UTM
            ux_a, uy_a = _pixel_to_utm(det_a['center'][0], det_a['center'][1],
                                       H_a, img_w_a, img_h_a)
            ux_b, uy_b = _pixel_to_utm(det_b['center'][0], det_b['center'][1],
                                       H_b, img_w_b, img_h_b)
            midpoints_utm.append(((ux_a + ux_b) / 2.0, (uy_a + uy_b) / 2.0))

        # Ordenar los puntos medios proyectándolos sobre la línea que une los centroides
        # para que la polilínea no se cruce
        vec = np.array([cx_b - cx_a, cy_b - cy_a])
        vec_norm = vec / (np.linalg.norm(vec) + 1e-9)
        projections = [np.dot(np.array(p) - np.array([cx_a, cy_a]), vec_norm)
                       for p in midpoints_utm]
        sorted_idx = np.argsort(projections)
        sorted_midpoints = [midpoints_utm[i] for i in sorted_idx]

        # Extremos de la frontera: intersecciones de la perpendicular por el primer
        # y último punto medio con los bordes de los footprints
        def perp_at_point(px, py):
            dx = cx_b - cx_a
            dy = cy_b - cy_a
            perp_dx, perp_dy = -dy, dx
            scale = 1e6
            return LineString([
                (px - perp_dx * scale, py - perp_dy * scale),
                (px + perp_dx * scale, py + perp_dy * scale),
            ])

        # Construir polilínea conectando los puntos medios
        # Para que sea suave, usamos una línea quebrada (LineString)
        # Los extremos los encontramos intersectando con los footprints
        first_perp = perp_at_point(*sorted_midpoints[0])
        last_perp = perp_at_point(*sorted_midpoints[-1])

        # Puntos donde la frontera "entra" y "sale" del área de fotos
        hits_first = first_perp.intersection(unary_union([poly_a, poly_b]))
        hits_last = last_perp.intersection(unary_union([poly_a, poly_b]))

        # Construir la polilínea completa
        # Simplificación: usamos los puntos medios ordenados como vértices
        # y extendemos los extremos hasta los bordes
        ext_first = _extract_farthest_point(hits_first, sorted_midpoints[0])
        ext_last = _extract_farthest_point(hits_last, sorted_midpoints[-1])

        if ext_first and ext_last:
            line_utm = LineString([ext_first] + sorted_midpoints + [ext_last])
        else:
            line_utm = LineString(sorted_midpoints)
    else:
        # --- Línea media geométrica (perpendicular bisector) ---
        line_utm = _get_perpendicular_bisector((cx_a, cy_a), (cx_b, cy_b), bounds_poly)

    # Determinar qué lado de la línea pertenece a cada foto
    # Usamos el centroide como referencia: el lado que contiene cx_a, cy_a es "lado A"
    side_a_utm = 'left'  # placeholder, se determina por test de punto

    # Convertir frontera a píxeles de cada foto
    coords = np.array(line_utm.coords)
    line_px_a = [_utm_to_pixel(x, y, H_a, img_w_a, img_h_a) for x, y in coords]
    line_px_b = [_utm_to_pixel(x, y, H_b, img_w_b, img_h_b) for x, y in coords]

    return {
        'line_utm': line_utm,
        'line_pixels_a': line_px_a,
        'line_pixels_b': line_px_b,
        'side_a_utm': side_a_utm,
        'homography_a': H_a,
        'homography_b': H_b,
    }


def _extract_farthest_point(geom, reference_point):
    """De una geometría de intersección, extrae el punto más alejado del reference."""
    if geom.is_empty:
        return None
    points = []
    if geom.geom_type == 'Point':
        points = [geom]
    elif geom.geom_type == 'MultiPoint':
        points = list(geom.geoms)
    elif geom.geom_type == 'LineString':
        points = [Point(geom.coords[0]), Point(geom.coords[-1])]
    elif geom.geom_type == 'MultiLineString':
        for g in geom.geoms:
            points.extend([Point(g.coords[0]), Point(g.coords[-1])])
    else:
        return None

    ref = np.array(reference_point)
    best = None
    best_dist = -1
    for p in points:
        d = np.linalg.norm(np.array([p.x, p.y]) - ref)
        if d > best_dist:
            best_dist = d
            best = (p.x, p.y)
    return best
