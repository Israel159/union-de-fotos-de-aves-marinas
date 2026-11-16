"""
Utilidades de visualización y conversión de coordenadas.
"""
import folium
from folium import Polygon as FoliumPolygon
import numpy as np
from pyproj import Transformer


def build_footprint_map(photos: list, pairs: list = None) -> folium.Map:
    """
    Crea un mapa Folium con los footprints de las fotos y, opcionalmente,
    resalta las zonas de superposición entre pares.
    """
    if not photos:
        return folium.Map(location=[0, 0], zoom_start=2)

    # Centro del mapa: promedio de lat/lon
    lats = [p['latitude'] for p in photos if p.get('latitude')]
    lons = [p['longitude'] for p in photos if p.get('longitude')]
    if not lats or not lons:
        return folium.Map(location=[0, 0], zoom_start=2)

    center = [np.mean(lats), np.mean(lons)]
    m = folium.Map(location=center, zoom_start=18, tiles='OpenStreetMap')

    # Añadir footprints
    for p in photos:
        fp = p.get('footprint')
        if not fp:
            continue
        # Convertir corners UTM a lat/lon para dibujar
        epsg = fp.get('epsg', 4326)
        transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
        corners = fp['corners']
        latlons = []
        for x, y in corners:
            lon, lat = transformer.transform(x, y)
            latlons.append([lat, lon])
        # Cerrar polígono
        latlons.append(latlons[0])

        FoliumPolygon(
            locations=latlons,
            color='#2563eb',
            weight=1.5,
            fill=True,
            fill_color='#3b82f6',
            fill_opacity=0.15,
            popup=f"Foto: {p.get('path', '').split('/')[-1]}<br>Alt: {p.get('altitude', 'N/A')}m"
        ).add_to(m)

    # Resaltar intersecciones
    if pairs:
        for pair in pairs:
            inter = pair.get('intersection_polygon')
            if not inter:
                continue
            # Asumimos mismo EPSG para ambas fotos del par
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
                    color='#dc2626',
                    weight=2,
                    fill=True,
                    fill_color='#ef4444',
                    fill_opacity=0.35,
                    popup=f"Superposición<br>{pair['intersection_area_m2']:.1f} m²"
                ).add_to(m)

    return m


def draw_boundary_on_image(img: 'Image.Image', line_pixels: list, color=(255, 0, 0), width=4):
    """Dibuja una línea de frontera sobre una imagen PIL."""
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    if len(line_pixels) >= 2:
        draw.line(line_pixels, fill=color, width=width)
    return img
