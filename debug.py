"""
Script de diagnóstico exhaustivo para "Union de Fotos de Aves Marinas".

Uso:
    python debug.py /ruta/a/carpeta/con/fotos

Genera un reporte completo con metadatos, footprints, pares candidatos,
y visualizaciones para detectar por qué no hay superposiciones.
"""
import sys
import os
import math
import json
import folium
import shapely
from folium import Polygon as FoliumPolygon
from datetime import datetime

from modules.exif_reader import read_metadata
from modules.footprint import batch_footprints, compute_footprint
from modules.overlap import find_overlapping_pairs


def _format_coord(val):
    return f"{val:.8f}" if val is not None else "N/A"


def _format_num(val, dec=2):
    return f"{val:.{dec}f}" if val is not None else "N/A"


def main(photos_dir: str, min_overlap: float = 1.0):
    print("=" * 80)
    print(f"DIAGNOSTICO: {photos_dir}")
    print(f"Fecha: {datetime.now().isoformat()}")
    print(f"Shapely: {shapely.__version__}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Encontrar fotos
    # ------------------------------------------------------------------
    exts = ('.jpg', '.jpeg', '.JPG', '.JPEG', '.tif', '.tiff', '.TIF', '.TIFF')
    files = sorted([os.path.join(photos_dir, f) for f in os.listdir(photos_dir)
                    if f.endswith(exts)])
    print(f"\n[1] Fotos encontradas: {len(files)}")
    if not files:
        print("ERROR: No se encontraron fotos.")
        return

    # ------------------------------------------------------------------
    # 2. Leer metadatos
    # ------------------------------------------------------------------
    print(f"\n[2] Leyendo metadatos EXIF/XMP...")
    metadata_list = [read_metadata(fp) for fp in files]

    print(f"\n{'FOTO':<30} {'LAT':>14} {'LON':>14} {'ALT':>8} {'F(mm)':>7} {'F35':>6} {'WxH':>12} {'XMP':>4}")
    print("-" * 95)
    valid_count = 0
    for m in metadata_list:
        has_gps = m.get('latitude') is not None and m.get('longitude') is not None
        if has_gps:
            valid_count += 1
        fname = os.path.basename(m['path'])
        print(f"{fname:<30} {_format_coord(m.get('latitude')):>14} "
              f"{_format_coord(m.get('longitude')):>14} "
              f"{_format_num(m.get('altitude'),1):>8} "
              f"{_format_num(m.get('focal_length'),1):>7} "
              f"{_format_num(m.get('focal_length_35mm'),0):>6} "
              f"{m.get('image_width')}x{m.get('image_height'):>5} "
              f"{'SI' if m.get('has_xmp') else 'NO':>4}")

    print(f"\nFotos con GPS valido: {valid_count}/{len(metadata_list)}")
    if valid_count == 0:
        print("\nERROR CRITICO: Ninguna foto tiene coordenadas GPS.")
        print("Posibles causas:")
        print("  - Las fotos fueron procesadas y perdieron el EXIF.")
        print("  - El dron no tenia GPS activado.")
        print("  - El formato XMP de DJI no se esta parseando bien.")
        return

    # ------------------------------------------------------------------
    # 3. Calcular footprints
    # ------------------------------------------------------------------
    print(f"\n[3] Calculando footprints...")
    photos = batch_footprints(metadata_list)

    print(f"\n{'FOTO':<30} {'AREA(m2)':>10} {'ANCHO':>8} {'ALTO':>8} {'EPSG':>8}")
    print("-" * 70)
    for p in photos:
        fp = p.get('footprint')
        fname = os.path.basename(p['path'])
        if fp:
            print(f"{fname:<30} {_format_num(fp.get('area_m2'),1):>10} "
                  f"{_format_num(fp.get('width_m'),1):>8} "
                  f"{_format_num(fp.get('height_m'),1):>8} "
                  f"{fp.get('epsg'):>8}")
        else:
            print(f"{fname:<30} NO SE PUDO CALCULAR FOOTPRINT")

    # Verificar EPSG homogéneo
    epsgs = set()
    for p in photos:
        fp = p.get('footprint')
        if fp:
            epsgs.add(fp.get('epsg'))
    print(f"\nZonas UTM encontradas: {sorted(epsgs)}")
    if len(epsgs) > 1:
        print("ADVERTENCIA: Las fotos cruzan zonas UTM diferentes!")
        print("Esto puede causar que las intersecciones no se detecten.")

    # ------------------------------------------------------------------
    # 4. Buscar pares con varios umbrales
    # ------------------------------------------------------------------
    print(f"\n[4] Buscando pares superpuestos (umbral actual: {min_overlap} m2)...")
    pairs = find_overlapping_pairs(photos, min_overlap_m2=min_overlap)
    print(f"Pares encontrados: {len(pairs)}")
    for i, pair in enumerate(pairs, start=1):
        name_a = os.path.basename(pair['photo_a']['path'])
        name_b = os.path.basename(pair['photo_b']['path'])
        if pair.get('intersection_polygon') is None:
            detail = f"fallback GPS={pair.get('fallback_gps_distance_m', 0):.2f} m"
        else:
            detail = f"intersección real={pair['intersection_area_m2']:.2f} m2"
        print(f"  {i}. {name_a} <-> {name_b}: {detail}")

    # Diagnosticar por qué 0: probar con umbral 0 y mostrar distancias
    print(f"\n[4.1] Diagnostico de distancias entre todos los pares...")
    print(f"{'FOTO_A':<25} {'FOTO_B':<25} {'DIST(m)':>10} {'AREA_INT':>10} {'STATUS':>20}")
    print("-" * 85)

    from shapely import unary_union
    from pyproj import Transformer

    for i in range(len(photos)):
        for j in range(i + 1, len(photos)):
            p_a = photos[i]
            p_b = photos[j]
            fp_a = p_a.get('footprint')
            fp_b = p_b.get('footprint')
            if not fp_a or not fp_b:
                continue

            # Distancia entre centroides (metros, ambos en UTM)
            cx_a, cy_a = fp_a['centroid_utm']
            cx_b, cy_b = fp_b['centroid_utm']
            dist = math.hypot(cx_a - cx_b, cy_a - cy_b)

            # Verificar EPSG
            if fp_a['epsg'] != fp_b['epsg']:
                status = "ZONAS UTM DIFERENTES"
                area = 0
            else:
                poly_a = fp_a['polygon']
                poly_b = fp_b['polygon']
                if poly_a.intersects(poly_b):
                    inter = poly_a.intersection(poly_b)
                    area = inter.area
                    if area >= min_overlap:
                        status = "SUPERPUESTO OK"
                    else:
                        status = f"AREA BAJA ({area:.1f} m2)"
                else:
                    area = 0
                    status = "NO INTERSECTAN"

            fname_a = os.path.basename(p_a['path'])
            fname_b = os.path.basename(p_b['path'])
            print(f"{fname_a:<25} {fname_b:<25} {_format_num(dist,1):>10} "
                  f"{_format_num(area,1):>10} {status:>20}")

    # ------------------------------------------------------------------
    # 5. Mapa de diagnóstico
    # ------------------------------------------------------------------
    print(f"\n[5] Generando mapa de diagnostico...")
    lats = [p['latitude'] for p in photos if p.get('latitude')]
    lons = [p['longitude'] for p in photos if p.get('longitude')]
    if lats and lons:
        center = [sum(lats) / len(lats), sum(lons) / len(lons)]
        m = folium.Map(location=center, zoom_start=18, tiles='OpenStreetMap')

        for p in photos:
            fp = p.get('footprint')
            if not fp:
                continue
            epsg = fp.get('epsg', 4326)
            transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
            corners = fp['corners']
            latlons = []
            for x, y in corners:
                lon, lat = transformer.transform(x, y)
                latlons.append([lat, lon])
            latlons.append(latlons[0])

            FoliumPolygon(
                locations=latlons,
                color='blue',
                weight=2,
                fill=True,
                fill_color='blue',
                fill_opacity=0.1,
                popup=f"{os.path.basename(p['path'])}<br>Alt: {p.get('altitude', 'N/A')}m"
            ).add_to(m)

        # Pintar intersecciones en rojo
        for pair in pairs:
            inter = pair.get('intersection_polygon')
            if not inter:
                continue
            epsg = pair['photo_a']['footprint']['epsg']
            transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
            coords = []
            if inter.geom_type == 'Polygon':
                coords = list(inter.exterior.coords)[:-1]
            elif inter.geom_type == 'MultiPolygon':
                coords = list(inter.geoms[0].exterior.coords)[:-1]
            latlons = []
            for x, y in coords:
                lon, lat = transformer.transform(x, y)
                latlons.append([lat, lon])
            if latlons:
                latlons.append(latlons[0])
                FoliumPolygon(
                    locations=latlons,
                    color='red',
                    weight=3,
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.4,
                    popup=f"Superposicion<br>{pair['intersection_area_m2']:.1f} m2"
                ).add_to(m)

        map_path = os.path.join(photos_dir, "debug_map.html")
        m.save(map_path)
        print(f"Mapa guardado en: {map_path}")
    else:
        print("No se pudo generar el mapa (sin coordenadas GPS).")

    # ------------------------------------------------------------------
    # 6. Guardar reporte JSON
    # ------------------------------------------------------------------
    report = {
        "fecha": datetime.now().isoformat(),
        "shapely_version": shapely.__version__,
        "directorio": photos_dir,
        "total_fotos": len(files),
        "fotos_con_gps": valid_count,
        "pares_superpuestos": len(pairs),
        "umbral_m2": min_overlap,
        "zonas_utm": sorted(epsgs),
        "fotos": [],
    }
    for p in photos:
        fp = p.get('footprint')
        report["fotos"].append({
            "archivo": os.path.basename(p['path']),
            "lat": p.get('latitude'),
            "lon": p.get('longitude'),
            "altitud": p.get('altitude'),
            "altitud_relativa": p.get('altitude_relative'),
            "altitud_abs": p.get('altitude_abs'),
            "fuente_altitud": p.get('altitude_source'),
            "advertencia_altitud": p.get('altitude_warning'),
            "focal_length": p.get('focal_length'),
            "focal_length_35mm": p.get('focal_length_35mm'),
            "dimensiones": [p.get('image_width'), p.get('image_height')],
            "has_xmp": p.get('has_xmp', False),
            "footprint": {
                "epsg": fp.get('epsg') if fp else None,
                "area_m2": fp.get('area_m2') if fp else None,
                "width_m": fp.get('width_m') if fp else None,
                "height_m": fp.get('height_m') if fp else None,
                "centroid_utm": fp.get('centroid_utm') if fp else None,
                "drone_utm": fp.get('drone_utm') if fp else None,
                "projection_method": fp.get('projection_method') if fp else None,
                "projection_warning": fp.get('projection_warning') if fp else None,
                "altitude_used_m": fp.get('altitude_used_m') if fp else None,
                "corners": fp.get('corners') if fp else None,
            }
        })

    json_path = os.path.join(photos_dir, "debug_report.json")
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Reporte JSON guardado en: {json_path}")

    print("\n" + "=" * 80)
    print("DIAGNOSTICO COMPLETO")
    print("=" * 80)
    print("\nPasos recomendados:")
    print("1. Abre el mapa (debug_map.html) en tu navegador.")
    print("2. Si los footprints NO se ven (o son muy pequenos), revisa la altitud.")
    print("3. Si los footprints estan lejos unos de otros, revisa las coordenadas GPS.")
    print("4. Si cruzan zonas UTM diferentes, las intersecciones pueden fallar.")
    print("5. Revisa el JSON para ver los valores exactos de cada foto.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python debug.py /ruta/a/carpeta/con/fotos")
        sys.exit(1)
    main(sys.argv[1])
