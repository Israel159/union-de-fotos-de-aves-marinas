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

    # Tamaño nadir nominal, también usado como referencia por el fallback GPS.
    ground_w = altitude * (sensor_w_mm / focal_length)
    ground_h = altitude * (sensor_h_mm / focal_length)

    yaw_rad = math.radians(gimbal_yaw if gimbal_yaw is not None else 0.0)
    pitch_rad = math.radians(gimbal_pitch if gimbal_pitch is not None else -90.0)
    roll_rad = math.radians(gimbal_roll if gimbal_roll is not None else 0.0)
    off_nadir = abs(pitch_rad + math.pi / 2)

    # Rectángulo de respaldo, centrado en el GPS del dron. El orden de las
    # esquinas coincide con boundary.py: abajo-izquierda, abajo-derecha,
    # arriba-derecha y arriba-izquierda de la imagen.
    hw, hh = ground_w / 2.0, ground_h / 2.0
    corners = np.array([
        [-hw, -hh],
        [hw, -hh],
        [hw, hh],
        [-hw, hh],
    ])
    rot = np.array([
        [math.cos(yaw_rad), -math.sin(yaw_rad)],
        [math.sin(yaw_rad), math.cos(yaw_rad)],
    ])
    corners_utm = (rot @ corners.T).T + np.array([cx, cy])
    center_utm = (cx, cy)
    projection_method = 'nadir_rectangular'
    projection_warning = None

    # Cámara inclinada: intersectar los rayos de las cuatro esquinas con un
    # plano de suelo horizontal. Coordenadas locales ENU: x=Este, y=Norte,
    # z=arriba. DJI yaw=0 apunta al Norte y pitch=-90 apunta a nadir.
    if off_nadir > math.radians(0.5) or abs(roll_rad) > math.radians(0.5):
        theta = pitch_rad + math.pi / 2  # 0=nadir; positivo hacia adelante
        forward = np.array([math.sin(yaw_rad), math.cos(yaw_rad), 0.0])
        right = np.array([math.cos(yaw_rad), -math.sin(yaw_rad), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        down_image = -math.cos(theta) * forward - math.sin(theta) * up
        optical = math.sin(theta) * forward - math.cos(theta) * up

        # Roll alrededor del eje óptico.
        right_roll = math.cos(roll_rad) * right + math.sin(roll_rad) * down_image
        down_roll = -math.sin(roll_rad) * right + math.cos(roll_rad) * down_image

        half_u = sensor_w_mm / (2.0 * focal_length)
        half_v = sensor_h_mm / (2.0 * focal_length)
        image_corners = [
            (-half_u, half_v),   # abajo-izquierda
            (half_u, half_v),    # abajo-derecha
            (half_u, -half_v),   # arriba-derecha
            (-half_u, -half_v),  # arriba-izquierda
        ]

        projected = []
        all_rays_hit_ground = True
        for u, v in image_corners:
            ray = u * right_roll + v * down_roll + optical
            if ray[2] >= -1e-8:
                all_rays_hit_ground = False
                break
            distance = -altitude / ray[2]
            east_north = np.array([cx, cy]) + distance * ray[:2]
            projected.append(east_north)

        if all_rays_hit_ground:
            candidate = Polygon(projected)
            if candidate.is_valid and candidate.area > 0:
                corners_utm = np.asarray(projected)
                center_distance = -altitude / optical[2]
                center_xy = np.array([cx, cy]) + center_distance * optical[:2]
                center_utm = (float(center_xy[0]), float(center_xy[1]))
                projection_method = 'oblique_ray_plane'
            else:
                projection_warning = (
                    'La proyección oblicua produjo un polígono inválido; '
                    'se usa el rectángulo nadir de respaldo.'
                )
        else:
            projection_warning = (
                'Al menos una esquina apunta al horizonte o por encima de él; '
                'no se puede intersectar con el suelo y se usa el rectángulo '
                'nadir de respaldo.'
            )

    poly = Polygon(corners_utm)
    return {
        'polygon': poly,
        'centroid_utm': center_utm,
        'drone_utm': (cx, cy),
        'epsg': epsg,
        'area_m2': poly.area,
        'width_m': ground_w,
        'height_m': ground_h,
        'corners': corners_utm.tolist(),
        'altitude_used_m': altitude,
        'off_nadir_deg': math.degrees(off_nadir),
        'projection_method': projection_method,
        'projection_warning': projection_warning,
        'terrain_model': 'flat_ground_approximation',
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
        fp['altitude_source'] = (
            meta.get('altitude_source')
            if meta.get('altitude') is not None and meta.get('altitude') > 0
            else 'default_50m'
        )
        meta_copy = dict(meta)
        meta_copy['footprint'] = fp
        results.append(meta_copy)
    return results
