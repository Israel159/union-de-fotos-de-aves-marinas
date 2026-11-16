"""
Detección genérica de aves con YOLOv8 y matching de duplicados entre fotos vecinas.
El modelo usado es el estándar de Ultralytics (COCO) que detecta la clase 'bird'.
"""
import math
import os
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple

# Importación diferida para no fallar en import si ultralytics no está instalado
try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except ImportError:
    _HAS_ULTRALYTICS = False
    YOLO = None

# Clase "bird" en el dataset COCO de YOLOv8
BIRD_CLASS_ID = 14


def load_yolo_model(model_name: str = "yolov8n.pt"):
    """Carga (o descarga) el modelo YOLO. Se cachea a nivel de proceso."""
    if not _HAS_ULTRALYTICS:
        raise RuntimeError("La librería 'ultralytics' no está instalada. Ejecuta: pip install ultralytics")
    return YOLO(model_name)


def detect_birds(image_path: str, model, conf_threshold: float = 0.25) -> List[dict]:
    """
    Corre el detector YOLO en una imagen y devuelve solo las detecciones de aves.

    Retorna lista de dicts:
        {
            'bbox': (x1, y1, x2, y2),  # en píxeles, coordenadas absolutas
            'confidence': float,
            'center': (cx, cy),        # centro del bbox en píxeles
            'class_id': int,
        }
    """
    if model is None:
        return []

    results = model(image_path, verbose=False)
    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls.item())
            if cls_id != BIRD_CLASS_ID:
                continue
            conf = float(box.conf.item())
            if conf < conf_threshold:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'center': (cx, cy),
                'class_id': cls_id,
            })
    return detections


def _pixel_to_utm(cx: float, cy: float, img_w: int, img_h: int,
                  footprint: dict) -> Tuple[float, float]:
    """
    Convierte coordenadas de píxeles (cx, cy) a coordenadas UTM aproximadas
    usando el footprint de la foto. Asume una transformación lineal dentro
    del polígono (aproximación válida para fotos con poca distorsión y nadir).
    """
    corners = np.array(footprint['corners'])  # 4x2 en orden [BL, BR, TR, TL]
    # Normalizar coordenadas de píxel a [0,1]
    nx = cx / img_w
    ny = 1.0 - (cy / img_h)  # invertir Y porque píxeles crecen hacia abajo

    # Interpolación bilineal simplificada entre las 4 esquinas
    # Orden de corners: inferior-izq, inferior-der, superior-der, superior-izq
    x = (
        corners[0][0] * (1 - nx) * (1 - ny) +
        corners[1][0] * nx * (1 - ny) +
        corners[2][0] * nx * ny +
        corners[3][0] * (1 - nx) * ny
    )
    y = (
        corners[0][1] * (1 - nx) * (1 - ny) +
        corners[1][1] * nx * (1 - ny) +
        corners[2][1] * nx * ny +
        corners[3][1] * (1 - nx) * ny
    )
    return float(x), float(y)


def match_duplicates(detections_a: List[dict], detections_b: List[dict],
                     photo_a: dict, photo_b: dict,
                     max_distance_m: float = 3.0) -> List[Tuple[dict, dict]]:
    """
    Empareja detecciones duplicadas entre dos fotos comparando la posición
    georreferenciada (UTM) de sus centros.

    Parámetros
    ----------
    detections_a, detections_b : list
        Salida de detect_birds() para cada foto.
    photo_a, photo_b : dict
        Metadatos con 'footprint' y 'image_width', 'image_height'.
    max_distance_m : float
        Distancia máxima en metros para considerar que dos detecciones son la misma ave.

    Retorna
    -------
    Lista de tuplas (det_a, det_b) con los pares emparejados.
    """
    fp_a = photo_a.get('footprint')
    fp_b = photo_b.get('footprint')
    if not fp_a or not fp_b:
        return []

    img_w_a = photo_a.get('image_width', 1)
    img_h_a = photo_a.get('image_height', 1)
    img_w_b = photo_b.get('image_width', 1)
    img_h_b = photo_b.get('image_height', 1)

    # Proyectar centros a UTM
    utm_a = []
    for d in detections_a:
        ux, uy = _pixel_to_utm(d['center'][0], d['center'][1], img_w_a, img_h_a, fp_a)
        utm_a.append((ux, uy, d))

    utm_b = []
    for d in detections_b:
        ux, uy = _pixel_to_utm(d['center'][0], d['center'][1], img_w_b, img_h_b, fp_b)
        utm_b.append((ux, uy, d))

    matched = []
    used_b = set()
    for ux_a, uy_a, det_a in utm_a:
        best_idx = -1
        best_dist = float('inf')
        for idx_b, (ux_b, uy_b, det_b) in enumerate(utm_b):
            if idx_b in used_b:
                continue
            dist = math.hypot(ux_a - ux_b, uy_a - uy_b)
            if dist < best_dist and dist <= max_distance_m:
                best_dist = dist
                best_idx = idx_b
        if best_idx != -1:
            matched.append((det_a, utm_b[best_idx][2]))
            used_b.add(best_idx)

    return matched
