"""Lectura de metadatos EXIF/XMP de fotos de dron (sin binarios externos).

Requiere solo Pillow (no necesita exiftool), ideal para Streamlit Cloud.
"""
import re
from PIL import Image

_XMP_YAW_PATTERNS = [
    rb'[Gg]imbal[Yy]aw[Dd]egree="(-?[\d.]+)"',
    rb'[Cc]amera[Yy]aw[Dd]egree="(-?[\d.]+)"',
    rb'[Ff]light[Yy]aw[Dd]egree="(-?[\d.]+)"',
]
_XMP_REL_ALT = rb'[Rr]elative[Aa]ltitude="(-?[\d.]+)"'
_XMP_ABS_ALT = rb'[Aa]bsolute[Aa]ltitude="(-?[\d.]+)"'


def _to_float(v):
    if v is None:
        return None
    try:
        if isinstance(v, tuple) and len(v) == 2:      # (num, den)
            return float(v[0]) / float(v[1]) if v[1] else None
        return float(v)                                # IFDRational, int, float
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _dms_to_dd(dms, ref):
    if not dms or len(dms) < 3:
        return None
    dd = sum(_to_float(x) / div for x, div in zip(dms, (1.0, 60.0, 3600.0)))
    if ref in ("S", "W"):
        dd = -dd
    return dd


def _read_xmp(path):
    """Devuelve (yaw_deg, altitud_m) desde el XMP de DJI, o (None, None)."""
    with open(path, "rb") as f:
        data = f.read()
    yaw = None
    for pat in _XMP_YAW_PATTERNS:
        m = re.search(pat, data)
        if m:
            yaw = float(m.group(1))
            break
    alt = None
    for pat in (_XMP_REL_ALT, _XMP_ABS_ALT):
        m = re.search(pat, data)
        if m:
            alt = float(m.group(1))
            break
    return yaw, alt


def extract_metadata(path):
    """Extrae lat, lon, alt_m, yaw_deg, img_w, img_h, focal_mm y tamaño de
    sensor derivado (sensor_w_mm/sensor_h_mm) desde EXIF + XMP de DJI."""
    img = Image.open(path)
    exif = img.getexd() if hasattr(img, "getexd") else img.getexif()
    gps = exif.get_ifd(0x8825) if exif else {}

    lat = _dms_to_dd(gps.get(2), gps.get(1))
    lon = _dms_to_dd(gps.get(4), gps.get(3))
    if lat is None or lon is None:
        raise ValueError("sin coordenadas GPS en EXIF")

    # Altitud: preferimos la relativa (sobre el terreno) del XMP de DJI.
    yaw, alt = _read_xmp(path)
    if alt is None:
        alt_ref = gps.get(5, 0)
        alt = _to_float(gps.get(6))
        if alt is not None and alt_ref == 1:
            alt = -alt
    if alt is None or alt <= 0:
        raise ValueError("sin altitud usable (revisa XMP/EXIF del dron)")

    focal = _to_float(exif.get(37386))          # FocalLength (real, mm)
    efl = _to_float(exif.get(41989))            # FocalLengthIn35mmFilm
    sensor_w = sensor_h = None
    if focal and efl:
        sensor_w = 36.0 * focal / efl           # ancho derivado del sensor
        sensor_h = 24.0 * focal / efl

    return {"lat": lat, "lon": lon, "alt_m": alt, "yaw_deg": yaw,
            "img_w": img.width, "img_h": img.height,
            "focal_mm": focal, "sensor_w_mm": sensor_w, "sensor_h_mm": sensor_h}
