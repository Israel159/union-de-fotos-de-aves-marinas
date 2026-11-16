"""
Lectura de metadatos EXIF y XMP (DJI) de fotografías de dron.
Extrae: GPS, altitud, orientación del gimbal y parámetros de la cámara.
"""
import struct
from io import BytesIO
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import xml.etree.ElementTree as ET


def _to_float(value):
    """Convierte un valor EXIF (int, float, str, IFDRational, tuple) a float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    # PIL IFDRational
    if hasattr(value, 'numerator') and hasattr(value, 'denominator'):
        return float(value.numerator) / float(value.denominator)
    # Si tiene __float__
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    # Tupla/lista tipo ratio
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return float(value[0]) / float(value[1])
    raise ValueError(f"No se puede convertir a float: {value} (tipo {type(value)})")


def _dms_to_dd(dms, ref):
    """Convierte grados/minutos/segundos a decimal."""
    if dms is None:
        return None
    d, m, s = dms
    dd = _to_float(d) + _to_float(m) / 60 + _to_float(s) / 3600
    if ref in ('S', 'W'):
        dd = -dd
    return dd


def _get_exif_value(exifdata, tag_name):
    """Extrae un valor del diccionario EXIF por nombre legible."""
    for tag_id, value in exifdata.items():
        if TAGS.get(tag_id) == tag_name:
            return value
    return None


def _get_gps_value(gps_info, tag_name):
    """Extrae un valor del diccionario GPSInfo por nombre legible."""
    for tag_id, value in gps_info.items():
        if GPSTAGS.get(tag_id) == tag_name:
            return value
    return None


def _extract_xmp(image_path: str) -> str:
    """Extrae el XML XMP crudo del segmento APP1 del JPEG."""
    with open(image_path, 'rb') as f:
        data = f.read()
    # Buscar el marcador APP1 (0xFFE1) seguido del namespace XMP
    idx = 0
    while True:
        idx = data.find(b'\xff\xe1', idx)
        if idx == -1:
            break
        # Longitud del segmento (big-endian, sin contar los 2 bytes del marcador)
        length = struct.unpack('>H', data[idx + 2:idx + 4])[0]
        segment = data[idx + 4:idx + 2 + length]
        if b'http://ns.adobe.com/xap/1.0/' in segment:
            # El contenido XMP empieza después del NULL que sigue al namespace
            null_pos = segment.find(b'\x00')
            if null_pos != -1:
                return segment[null_pos + 1:].decode('utf-8', errors='ignore')
        idx += 2
    return ""


def _parse_dji_xmp(xmp_xml: str) -> dict:
    """Parsea el XML XMP de DJI y extrae tags relevantes."""
    result = {}
    if not xmp_xml:
        return result
    try:
        root = ET.fromstring(xmp_xml.encode('utf-8'))
    except ET.ParseError:
        return result

    # Namespaces comunes de DJI
    ns = {
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'drone-dji': 'http://www.dji.com/drone-dji/1.0/',
        'x': 'adobe:ns:meta/',
    }

    # Buscar en Description
    for desc in root.iter('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description'):
        for key in [
            'GimbalYawDegree', 'GimbalPitchDegree', 'GimbalRollDegree',
            'FlightYawDegree', 'FlightPitchDegree', 'FlightRollDegree',
            'RelativeAltitude', 'AbsoluteAltitude',
            'GpsLatitude', 'GpsLongitude',
            'CalibratedFocalLength', 'RtkFlag',
        ]:
            # Intentar con namespace drone-dji
            val = desc.get(f'{{{ns["drone-dji"]}}}{key}')
            if val is not None:
                result[key] = val
    return result


def read_metadata(image_path: str) -> dict:
    """
    Lee metadatos EXIF y XMP de una imagen y devuelve un diccionario unificado.
    """
    meta = {
        'path': image_path,
        'latitude': None,
        'longitude': None,
        'altitude': None,           # altitud seleccionada para el cálculo
        'altitude_relative': None,  # XMP DJI RelativeAltitude, sin modificar
        'altitude_abs': None,       # metros absolutos / sobre el nivel del mar
        'altitude_source': None,    # relative, absolute_fallback o None
        'altitude_warning': None,
        'focal_length': None,       # mm (real del sensor)
        'focal_length_35mm': None,  # mm equivalente
        'image_width': None,
        'image_height': None,
        'gimbal_yaw': None,
        'gimbal_pitch': None,
        'gimbal_roll': None,
        'flight_yaw': None,
        'has_xmp': False,
    }

    # EXIF estándar vía PIL
    with Image.open(image_path) as img:
        meta['image_width'] = img.width
        meta['image_height'] = img.height
        exif = img._getexif()
        if exif:
            meta['focal_length'] = _get_exif_value(exif, 'FocalLength')
            meta['focal_length_35mm'] = _get_exif_value(exif, 'FocalLengthIn35mmFilm')

            gps_info = exif.get(34853)  # GPSInfo tag
            if gps_info:
                lat_dms = _get_gps_value(gps_info, 'GPSLatitude')
                lat_ref = _get_gps_value(gps_info, 'GPSLatitudeRef')
                lon_dms = _get_gps_value(gps_info, 'GPSLongitude')
                lon_ref = _get_gps_value(gps_info, 'GPSLongitudeRef')
                alt = _get_gps_value(gps_info, 'GPSAltitude')
                alt_ref = _get_gps_value(gps_info, 'GPSAltitudeRef')

                meta['latitude'] = _dms_to_dd(lat_dms, lat_ref)
                meta['longitude'] = _dms_to_dd(lon_dms, lon_ref)
                if alt is not None:
                    alt_val = _to_float(alt)
                    if alt_ref == 1:  # bajo el nivel del mar
                        alt_val = -alt_val
                    meta['altitude_abs'] = alt_val

    # XMP de DJI
    xmp_xml = _extract_xmp(image_path)
    if xmp_xml:
        meta['has_xmp'] = True
        dji = _parse_dji_xmp(xmp_xml)
        if dji.get('GpsLatitude'):
            meta['latitude'] = float(dji['GpsLatitude'])
        if dji.get('GpsLongitude'):
            meta['longitude'] = float(dji['GpsLongitude'])
        if dji.get('RelativeAltitude'):
            meta['altitude_relative'] = float(dji['RelativeAltitude'])
        if dji.get('AbsoluteAltitude'):
            meta['altitude_abs'] = float(dji['AbsoluteAltitude'])
        if dji.get('GimbalYawDegree'):
            meta['gimbal_yaw'] = float(dji['GimbalYawDegree'])
        if dji.get('GimbalPitchDegree'):
            meta['gimbal_pitch'] = float(dji['GimbalPitchDegree'])
        if dji.get('GimbalRollDegree'):
            meta['gimbal_roll'] = float(dji['GimbalRollDegree'])
        if dji.get('FlightYawDegree'):
            meta['flight_yaw'] = float(dji['FlightYawDegree'])

    # Seleccionar una altitud utilizable. RelativeAltitude puede ser cero o
    # negativa si el dron está por debajo del punto de despegue; no debe
    # convertirse silenciosamente en una altura fija de 50 m.
    relative = meta['altitude_relative']
    absolute = meta['altitude_abs']
    if relative is not None and relative > 0:
        meta['altitude'] = relative
        meta['altitude_source'] = 'relative'
    elif absolute is not None and absolute > 0:
        meta['altitude'] = absolute
        meta['altitude_source'] = 'absolute_fallback'
        if relative is not None and relative <= 0:
            meta['altitude_warning'] = (
                f'RelativeAltitude={relative:.3f} m no es válida como altura '
                f'sobre el suelo; se usa AbsoluteAltitude={absolute:.3f} m '
                'como aproximación.'
            )
    elif relative is not None:
        # Conservar el valor para diagnóstico; compute_footprint aplicará su
        # fallback explícito si no existe ninguna altitud positiva.
        meta['altitude'] = relative
        meta['altitude_source'] = 'invalid'
        meta['altitude_warning'] = (
            'No hay una altitud positiva en EXIF/XMP; el footprint será '
            'solamente una aproximación.'
        )

    # Normalizar focal_length y focal_length_35mm (pueden venir como IFDRational)
    if meta['focal_length'] is not None:
        meta['focal_length'] = _to_float(meta['focal_length'])
    if meta['focal_length_35mm'] is not None:
        meta['focal_length_35mm'] = _to_float(meta['focal_length_35mm'])

    return meta
