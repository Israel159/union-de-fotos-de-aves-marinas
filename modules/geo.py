"""Conversiones píxel <-> coordenadas geográficas.

Convenciones:
- yaw 0 = cámara apuntando al norte; yaw crece en sentido horario.
- En la imagen: derecha = este, abajo = sur (cuando yaw=0).
- Se asume vuelo casi nadir (cámara mirando al suelo).
"""
import math

def footprint_size_m(alt_m, focal_mm, sensor_mm):
    """Metros de suelo cubiertos según altura, focal y tamaño de sensor."""
    if not alt_m or alt_m <= 0:
        return None
    return alt_m * sensor_mm / focal_mm

def footprint_from_fov(alt_m, fov_deg):
    """Metros de suelo cubiertos a partir del FOV de la cámara."""
    if not alt_m or alt_m <= 0:
        return None
    return 2.0 * alt_m * math.tan(math.radians(fov_deg) / 2.0)

def pixel_to_ground_offset(px, py, img_w, img_h, g_w, g_h, yaw_deg=0.0):
    """Desplazamiento en metros (este, norte) del píxel respecto al centro
    de la huella, rotando por el yaw del dron."""
    r = (px - img_w / 2.0) / img_w * g_w      # componente "a la derecha" en el suelo
    u = -(py - img_h / 2.0) / img_h * g_h     # componente "arriba" en el suelo
    th = math.radians(yaw_deg)
    east = u * math.sin(th) + r * math.cos(th)
    north = u * math.cos(th) - r * math.sin(th)
    return east, north

def offset_to_latlon(lat, lon, east_m, north_m):
    """Suma un offset en metros a una coordenada (aprox. plana local)."""
    dlat = north_m / 111320.0
    dlon = east_m / (111320.0 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon

def bbox_center_and_ground_size(det, img_w, img_h, g_w, g_h):
    """Centro del bbox en píxeles y ancho aproximado del ave en metros."""
    x1, y1, x2, y2 = det["bbox"]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, (x2 - x1) / img_w * g_w
