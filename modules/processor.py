"""
Procesamiento de imágenes: creación de máscaras de recorte, pintado de blanco
y generación de lotes procesados para descarga.
"""
import io
import math
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict
import zipfile


def fit_split_line(midpoints_px: List[Tuple[float, float]],
                   img_w: int, img_h: int) -> List[Tuple[float, float]]:
    """
    Ajusta una línea recta a un conjunto de puntos (puntos medios de duplicados)
    y la extiende hasta los bordes de la imagen.

    Retorna [(x1,y1), (x2,y2)] con las coordenadas de los puntos de corte
    con los bordes de la imagen [0, img_w] x [0, img_h].
    """
    if len(midpoints_px) == 0:
        return None
    if len(midpoints_px) == 1:
        # Línea vertical por el único punto
        x0, y0 = midpoints_px[0]
        return [(x0, 0.0), (x0, float(img_h))]

    pts = np.array(midpoints_px)
    xs, ys = pts[:, 0], pts[:, 1]

    # Regresión lineal: y = m*x + b
    # Si la línea es más vertical que horizontal, invertimos (x = m*y + b)
    dx = xs.max() - xs.min()
    dy = ys.max() - ys.min()

    if dx > dy:
        # Ajuste y = m*x + b
        A = np.vstack([xs, np.ones(len(xs))]).T
        m, b = np.linalg.lstsq(A, ys, rcond=None)[0]
        # Cortes con bordes x=0 y x=img_w
        y_at_0 = b
        y_at_w = m * img_w + b
        return [(0.0, float(y_at_0)), (float(img_w), float(y_at_w))]
    else:
        # Ajuste x = m*y + b
        A = np.vstack([ys, np.ones(len(ys))]).T
        m, b = np.linalg.lstsq(A, xs, rcond=None)[0]
        x_at_0 = b
        x_at_h = m * img_h + b
        return [(float(x_at_0), 0.0), (float(x_at_h), float(img_h))]


def _line_side(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Valor del semiplano definido por la línea (x1,y1)->(x2,y2).
    > 0 indica un lado, < 0 el otro.
    """
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def create_mask(img_w: int, img_h: int,
                line_points: List[Tuple[float, float]],
                reference_point: Tuple[float, float]) -> np.ndarray:
    """
    Crea una máscara booleana (True = pintar de blanco) para los píxeles
    que están del mismo lado de la línea que reference_point.

    Parámetros
    ----------
    img_w, img_h : int
        Dimensiones de la imagen.
    line_points : [(x1,y1), (x2,y2)]
        Dos puntos que definen la línea de frontera (en coordenadas de píxel).
    reference_point : (rx, ry)
        Un punto de referencia cuyo lado se marcará como True.

    Retorna
    -------
    np.ndarray de shape (img_h, img_w) con dtype bool.
    """
    if line_points is None or len(line_points) < 2:
        return np.zeros((img_h, img_w), dtype=bool)

    x1, y1 = line_points[0]
    x2, y2 = line_points[1]
    rx, ry = reference_point

    ref_val = _line_side(rx, ry, x1, y1, x2, y2)
    ref_sign = 1 if ref_val >= 0 else -1

    # Vectorizar con malla de coordenadas
    yy, xx = np.mgrid[0:img_h, 0:img_w]
    vals = (x2 - x1) * (yy - y1) - (y2 - y1) * (xx - x1)
    mask = (vals * ref_sign) >= 0
    return mask


def apply_white_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """
    Aplica una máscara booleana a una imagen PIL: los píxeles True se pintan de blanco.
    """
    arr = np.array(image)
    if len(arr.shape) == 2:
        # Escala de grises
        arr[mask] = 255
    else:
        # RGB/RGBA
        arr[mask] = 255
    return Image.fromarray(arr)


def process_photo_pair(photo_a: dict, photo_b: dict,
                       boundary_result: dict,
                       detections_a: List[dict], detections_b: List[dict],
                       matched_duplicates: List[Tuple[dict, dict]],
                       use_detector: bool = True) -> Tuple[Image.Image, Image.Image]:
    """
    Procesa un par de fotos superpuestas aplicando la frontera calculada.
    Pinta de blanco la parte de la superposición que corresponde a la otra foto.

    Retorna (img_a_processed, img_b_processed).
    """
    img_a = Image.open(photo_a['path']).convert('RGB')
    img_b = Image.open(photo_b['path']).convert('RGB')

    H_a = boundary_result['homography_a']
    H_b = boundary_result['homography_b']
    img_w_a, img_h_a = img_a.size
    img_w_b, img_h_b = img_b.size

    # Calcular frontera en píxeles
    if use_detector and len(matched_duplicates) > 0:
        # Puntos medios de duplicados en píxeles de cada foto
        mids_a = []
        mids_b = []
        for det_a, det_b in matched_duplicates:
            mids_a.append(det_a['center'])
            mids_b.append(det_b['center'])
        line_a = fit_split_line(mids_a, img_w_a, img_h_a)
        line_b = fit_split_line(mids_b, img_w_b, img_h_b)
    else:
        # Línea media geométrica: perpendicular bisector proyectado
        # Proyectamos el punto medio de los centroides UTM a píxeles
        from modules.boundary import _utm_to_pixel
        cx_a, cy_a = photo_a['footprint']['centroid_utm']
        cx_b, cy_b = photo_b['footprint']['centroid_utm']
        mx, my = (cx_a + cx_b) / 2.0, (cy_a + cy_b) / 2.0
        mpx_a, mpy_a = _utm_to_pixel(mx, my, H_a, img_w_a, img_h_a)
        mpx_b, mpy_b = _utm_to_pixel(mx, my, H_b, img_w_b, img_h_b)

        # Vector de A a B en UTM
        dx = cx_b - cx_a
        dy = cy_b - cy_a
        # Perpendicular
        perp_utm = [(-dy, dx), (dy, -dx)]
        # Extender desde el punto medio en ambas direcciones
        scale = 1e6
        p1_utm = (mx - dy * scale, my + dx * scale)
        p2_utm = (mx + dy * scale, my - dx * scale)
        line_a = [
            _utm_to_pixel(p1_utm[0], p1_utm[1], H_a, img_w_a, img_h_a),
            _utm_to_pixel(p2_utm[0], p2_utm[1], H_a, img_w_a, img_h_a),
        ]
        line_b = [
            _utm_to_pixel(p1_utm[0], p1_utm[1], H_b, img_w_b, img_h_b),
            _utm_to_pixel(p2_utm[0], p2_utm[1], H_b, img_w_b, img_h_b),
        ]

    # Puntos de referencia: centroide de la otra foto proyectado
    from modules.boundary import _utm_to_pixel
    ref_a = _utm_to_pixel(cx_b, cy_b, H_a, img_w_a, img_h_a)
    ref_b = _utm_to_pixel(cx_a, cy_a, H_b, img_w_b, img_h_b)

    mask_a = create_mask(img_w_a, img_h_a, line_a, ref_a)
    mask_b = create_mask(img_w_b, img_h_b, line_b, ref_b)

    img_a_proc = apply_white_mask(img_a, mask_a)
    img_b_proc = apply_white_mask(img_b, mask_b)

    return img_a_proc, img_b_proc


def build_download_zip(processed_images: Dict[str, Image.Image]) -> bytes:
    """
    Empaqueta un dict {nombre_archivo: PIL Image} en un ZIP en memoria.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname, img in processed_images.items():
            img_buf = io.BytesIO()
            img.save(img_buf, format='JPEG', quality=95)
            zf.writestr(fname, img_buf.getvalue())
    buf.seek(0)
    return buf.read()
