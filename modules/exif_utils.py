"""Lectura de metadatos EXIF/XMP de fotos de dron (sin binarios externos).

Soporta JPG/JPEG y TIFF. Requiere solo Pillow (no necesita exiftool), ideal
para Streamlit Cloud.

Correcciones respecto a la versión anterior:
1. Las altitudes XMP de DJI llevan signo '+' (p.ej. RelativeAltitude="+47.10");
   el regex antiguo no aceptaba '+' y caía silenciosamente en la altitud GPS.
2. Una foto con GPS pero sin altitud ya NO se descarta: devuelve alt_m=None y
   la app aplica la "altitud por defecto" configurada en la barra lateral.
3. Se busca altitud en más fuentes XMP (GpsAltitude de otros fabricantes).
"""
import re
from PIL import Image

_XMP_YAW_PATTERNS = [
    rb'[Gg]imbal[Yy]aw[Dd]egree="(-?[\d.]+)"',
    rb'[Cc]amera[Yy]aw[Dd]egree="(-?[\d.]+)"',
    rb'[Ff]light[Yy]aw[Dd]egree="(-?[\d.]+)"',
    rb'[Yy]aw[Dd]egree="(-?[\d.]+)"',
]
# Orden de preferencia: relativa > absoluta > GpsAltitude. El signo '+' es
# obligatorio en el regex: DJI escribe "+47.10".
_XMP_ALT_PATTERNS = [
    rb'[Rr]elative[Aa]ltitude="([+-]?[\d.]+)"',
    rb'[Aa]bsolute[Aa]ltitude="([+-]?[\d.]+)"',
    rb'[Gg]ps[Aa]ltitude="([+-]?[\d.]+)"',
]


def _to_float(v):
    if v is None:
        return None
    try:
        if isinstance(v, tuple) and len(v) == 2:      # (num, den)
            return float(v[0]) / float(v[1]) if v[1] else None
        return float(v)                                # IFDRational, int, float, str
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _dms_to_dd(dms, ref):
    if not dms or len(dms) < 3:
        return None
    dd = sum(_to_float(x) / div for x, div in zip(dms, (1.0, 60.0, 3600.0)))
    if ref in ("S", "W"):
        dd = -dd
    return dd


def _first_xmp_float(data, patterns):
    for pat in patterns:
        m = re.search(pat, data)
        if m:
            return float(m.group(1))
    return None


def _read_xmp(path):
    """Devuelve (yaw_deg, altitud_m) desde el XMP del dron, o (None, None)."""
    with open(path, "rb") as f:
        data = f.read()
    return (_first_xmp_float(data, _XMP_YAW_PATTERNS),
            _first_xmp_float(data, _XMP_ALT_PATTERNS))


def extract_metadata(path):
    """Extrae lat, lon, alt_m, yaw_deg, img_w, img_h, focal_mm y tamaño de
    sensor derivado (sensor_w_mm/sensor_h_mm) desde EXIF + XMP.

    alt_m puede ser None si la foto tiene GPS pero ninguna fuente de altitud;
    en ese caso la app debe aplicar una altitud por defecto.
    Lanza ValueError solo si no hay coordenadas GPS."""
    img = Image.open(path)
    exif = img.getexd() if hasattr(img, "getexd") else img.getexif()
    gps = exif.get_ifd(0x8825) if exif else {}

    lat = _dms_to_dd(gps.get(2), gps.get(1))
    lon = _dms_to_dd(gps.get(4), gps.get(3))
    if lat is None or lon is None:
        raise ValueError(
            "sin coordenadas GPS en EXIF. Ojo: los compresores de imágenes de "
            "internet eliminan los metadatos EXIF/XMP. Usa el JPG original "
            "que bajaste de la tarjeta del dron (o comprime sin borrar EXIF).")

    # Altitud: preferimos la relativa (sobre el terreno) del XMP del dron.
    yaw, alt = _read_xmp(path)
    if alt is None:
        alt_ref = gps.get(5, 0)
        alt = _to_float(gps.get(6))
        if alt is not None and alt_ref == 1:
            alt = -alt
        if alt is not None and alt <= 0:
            alt = None  # altitud GPS inválida/ausente -> que decida la app

    focal = _to_float(exif.get(37386))          # FocalLength (real, mm)
    efl = _to_float(exif.get(41989))            # FocalLengthIn35mmFilm
    sensor_w = sensor_h = None
    if focal and efl:
        sensor_w = 36.0 * focal / efl           # ancho derivado del sensor
        sensor_h = 24.0 * focal / efl

    return {"lat": lat, "lon": lon, "alt_m": alt, "yaw_deg": yaw,
            "img_w": img.width, "img_h": img.height,
            "focal_mm": focal, "sensor_w_mm": sensor_w, "sensor_h_mm": sensor_h}
