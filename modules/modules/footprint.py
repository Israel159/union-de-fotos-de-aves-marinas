"""
Cálculo del footprint georreferenciado de fotos de dron en el suelo.
Convierte lat/lon + altitud + orientación + parámetros de cámara
en un polígono (shapely) representando el área fotografiada.
"""
import math
import numpy as np
from pyproj import Transformer, CRS
from shapely.geometry import Polygon
from shapely.affinity import rotate


def _get_utm_epsg(lat: float, lon: float) -> int:
    """Devuelve el código EPSG del UTM zone apropiado."""
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone
    else:
        return 32700 + zone


def _compute_sensor_size(focal_length: float, focal_35mm: float = None,
                         default_w: float = 6.17, default_h: float = 4.55) -> tuple:
    """
    Calcula el tamaño físico del sensor en mm.
    Si se dispone de focal equivalente en 35mm, se deriva el crop factor.
    Si no, se usan valores por defecto (Phantom 3 Pro: 6.17 x 4.55 mm).
    """
    if focal_length and focal_35mm and focal_35mm > 0 and focal_length > 0:
        crop = focal_35mm / focal_length
        sensor_w = 36.0 / crop
        sensor_h = 24.0 / crop
        return sensor_w, sensor_h
    return default_w, default_h


def compute_footprint(lat: float, lon: float, altitude: float,
                      focal_length: float, image_w: int, image_h: int,
                      focal_35mm: float = None,
                      gimbal_yaw: float = 0.0,
                      gimbal_pitch: float = -90.0,
                      gimbal_roll: float = 0.0,
                      sensor_w_mm: float = None,
                      sensor_h_mm: float = None) -> dict:
    """
    Calcula el footprint de una foto de dron.

    Parámetros
    ----------
    lat, lon : float
        Coordenadas del centro de la imagen (dónde apunta la cámara).
    altitude : float
        Altura sobre el suelo en metros (AGL).
    focal_length : float
        Longitud focal real del sensor en milímetros.
    image_w, image_h : int
        Dimensiones de la imagen en píxeles.
    focal_35mm : float, opcional
        Longitud focal equivalente en 35mm.
    gimbal_yaw, gimbal_pitch, gimbal_roll : float
        Orientación del gimbal en grados.
        yaw=0 significa apuntando al Norte, pitch=-90 es nadir (vertical).
    sensor_w_mm, sensor_h_mm : float, opcional
        Tamaño físico del sensor. Si no se da, se calcula automáticamente.

    Retorna
    -------
    dict con:
        'polygon' : shapely Polygon (coordenadas UTM en metros, origen relativo)
        'centroid_utm' : (x, y) en UTM
        'epsg' : código EPSG del UTM usado
        'area_m2' : área aproximada en m²
        'width_m' : ancho aproximado en m
        'height_m' : alto aproximado en m
    """
    if altitude is None or altitude <= 0:
        altitude = 50.0  # fallback razonable para aves marinas

    if sensor_w_mm is None or sensor_h_mm is None:
        sensor_w_mm, sensor_h_mm = _compute_sensor_size(focal_length, focal_35mm)

    # Convertir centro a UTM
    epsg = _get_utm_epsg(lat, lon)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    cx, cy = transformer.transform(lon, lat)

    # --- Caso simplificado: nadir (pitch ≈ -90°, roll ≈ 0°) ---
    # El footprint es aproximadamente un rectángulo centrado en (cx, cy)
    # Las dimensiones en suelo se derivan por triángulos semejantes:
    #   ground_size = altitude * sensor_size / focal_length
    ground_w = altitude * (sensor_w_mm / focal_length)
    ground_h = altitude * (sensor_h_mm / focal_length)

    # Esquinas del rectángulo sin rotar, centradas en origen
    hw, hh = ground_w / 2.0, ground_h / 2.0
    corners = np.array([
        [-hw, -hh],
        [ hw, -hh],
        [ hw,  hh],
        [-hw,  hh],
    ])

    # Aplicar yaw (rotación alrededor del eje Z)
    yaw_rad = math.radians(gimbal_yaw if gimbal_yaw is not None else 0.0)
    rot = np.array([
        [math.cos(yaw_rad), -math.sin(yaw_rad)],
        [math.sin(yaw_rad),  math.cos(yaw_rad)],
    ])
    corners_rot = (rot @ corners.T).T

    # Trasladar al centro UTM
    corners_utm = corners_rot + np.array([cx, cy])

    # Pitch / roll no exactamente nadir: proyectar con homografía simplificada
    # Si pitch está lejos de -90° o roll != 0, aplicamos una corrección de perspectiva
    # aproximada escalando los lados frontales/traseros.
    pitch_rad = math.radians(gimbal_pitch if gimbal_pitch is not None else -90.0)
    roll_rad = math.radians(gimbal_roll if gimbal_roll is not None else 0.0)

    # Solo corregimos si hay inclinación significativa (> 5° de nadir o roll)
    if abs(pitch_rad + math.pi / 2) > 0.087 or abs(roll_rad) > 0.087:
        # Ángulo de inclinación respecto a nadir
        theta_p = -(pitch_rad + math.pi / 2)  # 0 cuando es nadir
        theta_r = roll_rad

        # Factor de escala por inclinación (aproximación de perspectiva)
        # Para vuelos de dron estándar de aves, la inclinación suele ser pequeña
        # así que esta aproximación lineal es suficiente.
        fx = 1 / max(math.cos(theta_p), 0.1)
        fy = 1 / max(math.cos(theta_r), 0.1)

        # Recomputar con escalas
        corners_p = np.array([
            [-hw * fx, -hh * fy],
            [ hw * fx, -hh * fy],
            [ hw * fx,  hh * fy],
            [-hw * fx,  hh * fy],
        ])
        corners_rot = (rot @ corners_p.T).T
        corners_utm = corners_rot + np.array([cx, cy])

    poly = Polygon(corners_utm)
    return {
        'polygon': poly,
        'centroid_utm': (cx, cy),
        'epsg': epsg,
        'area_m2': poly.area,
        'width_m': ground_w,
        'height_m': ground_h,
        'corners': corners_utm.tolist(),
    }


def batch_footprints(metadata_list: list) -> list:
    """
    Procesa una lista de metadatos y devuelve la lista enriquecida con footprints.
    Cada elemento de metadata_list debe ser un dict con las claves esperadas
    por compute_footprint.
    """
    results = []
    for meta in metadata_list:
        fp = compute_footprint(
            lat=meta.get('latitude'),
            lon=meta.get('longitude'),
            altitude=meta.get('altitude'),
            focal_length=meta.get('focal_length'),
            image_w=meta.get('image_width'),
            image_h=meta.get('image_height'),
            focal_35mm=meta.get('focal_length_35mm'),
            gimbal_yaw=meta.get('gimbal_yaw'),
            gimbal_pitch=meta.get('gimbal_pitch'),
            gimbal_roll=meta.get('gimbal_roll'),
        )
        meta_copy = dict(meta)
        meta_copy['footprint'] = fp
        results.append(meta_copy)
    return results
