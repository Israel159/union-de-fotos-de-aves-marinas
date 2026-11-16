"""
Union de Fotos de Aves Marinas — Streamlit App
================================================
Aplicación para eliminar superposiciones entre fotos de dron tomando
en cuenta la georreferencia EXIF/XMP y detecciones genéricas de aves (YOLOv8).

Flujo:
1. Subir lote de fotos (JPEG con EXIF/XMP de DJI).
2. Leer metadatos, calcular footprints, detectar pares superpuestos.
3. Para cada par: proponer frontera (geométrica o híbrida con detecciones).
4. Permitir edición manual de la frontera (offset + rotación).
5. Aplicar máscaras y descargar el lote procesado.
"""
import os
import io
import math
import tempfile
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Unión de Fotos de Aves Marinas",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Importación de módulos propios
# ---------------------------------------------------------------------------
from modules.exif_reader import read_metadata
from modules.footprint import batch_footprints
from modules.overlap import find_overlapping_pairs
from modules.detector import load_yolo_model, detect_birds, match_duplicates
from modules.boundary import compute_boundary, _utm_to_pixel
from modules.processor import (
    fit_split_line, create_mask, apply_white_mask,
    build_download_zip,
)
from modules.utils import build_footprint_map, draw_boundary_on_image

# ---------------------------------------------------------------------------
# Cachés
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo YOLOv8...")
def cached_yolo(model_name: str = "yolov8n.pt"):
    return load_yolo_model(model_name)


@st.cache_data(show_spinner="Leyendo metadatos EXIF/XMP...")
def cached_metadata(file_paths: tuple) -> list:
    """Lee metadatos de una lista de rutas de archivos."""
    return [read_metadata(fp) for fp in file_paths]


def cached_footprints(metadata_tuple: tuple) -> list:
    """Calcula footprints a partir de metadatos.
    Nota: no usamos @st.cache_data porque el resultado contiene objetos
    shapely.Polygon que no son hashables y provocan UnhashableParamError.
    """
    return batch_footprints(list(metadata_tuple))


def cached_pairs(photos_tuple: tuple, min_overlap: float, use_fallback: bool) -> list:
    """Detecta pares superpuestos.
    Nota: no usamos @st.cache_data porque la entrada contiene objetos
    shapely.Polygon que no son hashables.
    """
    return find_overlapping_pairs(
        list(photos_tuple),
        min_overlap_m2=min_overlap,
        use_fallback=use_fallback,
    )


# ---------------------------------------------------------------------------
# Sidebar — Configuración y carga
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuración")

uploaded_files = st.sidebar.file_uploader(
    "📤 Sube tus fotos de dron (JPEG / TIFF)",
    type=["jpg", "jpeg", "JPG", "JPEG", "tif", "tiff", "TIF", "TIFF"],
    accept_multiple_files=True,
)

model_name = st.sidebar.text_input(
    "Modelo YOLO para frontera",
    value="yolov8n.pt",
    help="Nombre del archivo .pt dentro de la carpeta models/. "
         "Puedes cambiarlo por yolov8s.pt, yolov8m.pt, etc. "
         "Si no existe, se descargará automáticamente.",
)

use_detector = st.sidebar.checkbox(
    "Usar detector de aves (YOLOv8) para refinar frontera",
    value=True,
    help="Si está activado, la frontera se ajusta pasando entre aves duplicadas. "
         "Si no, usa solo la línea media geométrica basada en GPS.",
)

class_id = st.sidebar.number_input(
    "ID de clase a detectar",
    min_value=-1, max_value=999, value=14,
    help="ID de la clase que representa 'ave' en tu modelo. "
         "Modelo COCO genérico (bird): 14. "
         "Modelo custom con una sola clase (ave): 0. "
         "Usa -1 para detectar TODAS las clases.",
)

conf_threshold = st.sidebar.slider(
    "Umbral de confianza del detector", 0.05, 0.9, 0.25, 0.05,
    help="Confianza mínima para considerar una detección válida.",
)

tile_size = st.sidebar.selectbox(
    "Tamaño de tile (tiling nativo)",
    options=[None, 320, 480, 640, 1024, 1280, 1536],
    index=0,
    format_func=lambda x: "Sin tiling (inferencia directa)" if x is None else f"{x} px",
    help="Para fotos de alta resolución (dron), usar un tile más pequeño "
         "permite detectar aves más pequeñas. Activa el tiling nativo de Ultralytics "
         "(reemplaza a SAHI). Recomendado: 640 o 1024 para fotos 4K.",
)

max_match_distance = st.sidebar.slider(
    "Distancia máxima para emparejar duplicados (m)", 0.5, 10.0, 3.0, 0.5,
    help="Dos detecciones en fotos distintas se consideran la misma ave "
         "si sus centros están a menos de esta distancia en metros.",
)

min_overlap_area = st.sidebar.slider(
    "Área mínima de superposición (m²)", 0.0, 100.0, 1.0, 1.0,
    help="Pares con intersección menor a este valor se ignoran.",
)

st.sidebar.markdown("---")

use_fallback = st.sidebar.checkbox(
    "Fallback por distancia GPS",
    value=True,
    help="Si no hay pares por footprints, busca pares cercanos usando "
         "distancia directa entre coordenadas GPS (Haversine). "
         "Util cuando las coordenadas GPS son imprecisas o el footprint se calcula mal.",
)

debug_mode = st.sidebar.checkbox(
    "Modo debug (mostrar diagnostico)",
    value=False,
    help="Muestra tablas con metadatos, footprints y distancias entre pares.",
)

st.sidebar.info(
    "**Nota:** El detector usado aquí es solo para calcular la frontera entre fotos. "
    "Tu modelo de especies (pinguino_pelicano_piquero.pt) se usaría en un paso posterior, "
    "sobre las fotos ya limpias."
)

# ---------------------------------------------------------------------------
# Estado de sesión
# ---------------------------------------------------------------------------
if "pair_adjustments" not in st.session_state:
    st.session_state.pair_adjustments = {}   # pair_idx -> {"offset": float, "angle": float}
if "processed_ready" not in st.session_state:
    st.session_state.processed_ready = False
if "processed_images" not in st.session_state:
    st.session_state.processed_images = {}

# ---------------------------------------------------------------------------
# Procesamiento principal (se ejecuta cuando hay archivos subidos)
# ---------------------------------------------------------------------------
if not uploaded_files:
    st.title("🐦 Unión de Fotos de Aves Marinas")
    st.markdown(
        """
        Bienvenido. Esta aplicación te permite:
        1. **Subir un lote de fotos** tomadas con dron (con georreferencia EXIF/XMP).
        2. **Detectar automáticamente** qué fotos se superponen.
        3. **Dibujar una frontera** entre cada par superpuesto para evitar contar el mismo ave dos veces.
        4. **Editar la frontera** manualmente si el algoritmo se equivoca.
        5. **Descargar las fotos procesadas** con la zona superpuesta pintada de blanco.

        👈 Comienza subiendo tus fotos en el panel de la izquierda.
        """
    )
    st.stop()

# Guardar archivos subidos a directorio persistente (evita que se borren entre reruns)
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "union_fotos_aves_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

file_paths = []
for uf in uploaded_files:
    path = os.path.join(UPLOAD_DIR, uf.name)
    with open(path, "wb") as f:
        f.write(uf.getvalue())
    file_paths.append(path)

# --- Paso 1: Metadatos y footprints ---
with st.spinner("Leyendo metadatos EXIF/XMP..."):
    metadata_list = cached_metadata(tuple(file_paths))

# Filtrar fotos sin GPS
valid_photos = [
    m for m in metadata_list
    if m.get('latitude') is not None and m.get('longitude') is not None
]
if not valid_photos:
    st.error("Ninguna de las fotos subidas contiene coordenadas GPS válidas. "
             "Asegúrate de que sean fotos de dron con metadatos intactos.")
    st.stop()

with st.spinner("Calculando footprints..."):
    photos = cached_footprints(tuple(valid_photos))

with st.spinner("Buscando pares superpuestos..."):
    pairs = cached_pairs(tuple(photos), min_overlap_area, use_fallback)

st.title("🐦 Unión de Fotos de Aves Marinas")

# Verificar si se usó fallback
fallback_count = sum(1 for p in pairs if p.get('intersection_polygon') is None)
if fallback_count > 0:
    st.warning(f"Se encontraron {len(pairs)} par(es). {fallback_count} par(es) detectados por fallback GPS "
               f"(los footprints no intersectaban). Revisa el mapa y las fronteras.")
else:
    st.success(f"Se encontraron {len(pairs)} par(es) superpuesto(s) de {len(photos)} foto(s).")


# ---------------------------------------------------------------------------
# Modo debug: información de diagnóstico
# ---------------------------------------------------------------------------
if debug_mode:
    with st.expander("🔍 Diagnostico (modo debug)", expanded=True):
        st.subheader("Metadatos de fotos")
        debug_data = []
        for p in photos:
            fp = p.get('footprint')
            debug_data.append({
                "Foto": os.path.basename(p['path']),
                "Lat": p.get('latitude'),
                "Lon": p.get('longitude'),
                "Alt relativa (m)": p.get('altitude_relative'),
                "Alt absoluta (m)": p.get('altitude_abs'),
                "Alt usada (m)": fp.get('altitude_used_m') if fp else p.get('altitude'),
                "Fuente altitud": fp.get('altitude_source') if fp else None,
                "Pitch gimbal (°)": p.get('gimbal_pitch'),
                "Proyección": fp.get('projection_method') if fp else None,
                "Focal (mm)": p.get('focal_length'),
                "Focal35 (mm)": p.get('focal_length_35mm'),
                "Dimensiones": f"{p.get('image_width')}x{p.get('image_height')}",
                "XMP": "SI" if p.get('has_xmp') else "NO",
                "Footprint W (m)": round(fp.get('width_m'), 1) if fp else None,
                "Footprint H (m)": round(fp.get('height_m'), 1) if fp else None,
                "EPSG": fp.get('epsg') if fp else None,
            })
        st.dataframe(debug_data, use_container_width=True)

        for p in photos:
            warnings = [
                w for w in (
                    p.get('altitude_warning'),
                    p.get('footprint', {}).get('projection_warning'),
                ) if w
            ]
            for warning in warnings:
                st.warning(f"{os.path.basename(p['path'])}: {warning}")

        if pairs:
            st.subheader("Detalle de pares")
            pair_data = []
            for i, pair in enumerate(pairs):
                is_fallback = pair.get('intersection_polygon') is None
                dist = pair.get('fallback_gps_distance_m')
                pair_data.append({
                    "Par": i + 1,
                    "Foto A": os.path.basename(pair['photo_a']['path']),
                    "Foto B": os.path.basename(pair['photo_b']['path']),
                    "Tipo": "Fallback GPS" if is_fallback else "Footprint",
                    "Area (m2)": round(pair.get('intersection_area_m2', 0), 1),
                    "Dist GPS (m)": round(dist, 1) if dist is not None else None,
                })
            st.dataframe(pair_data, use_container_width=True)
        else:
            st.error("No se encontraron pares. Revisa los metadatos arriba.")
            # Mostrar distancias entre pares para diagnosticar
            import math as _math
            st.subheader("Distancias entre todos los pares (Haversine)")
            dist_data = []
            for i in range(len(photos)):
                for j in range(i + 1, len(photos)):
                    p_a = photos[i]
                    p_b = photos[j]
                    lat_a, lon_a = p_a.get('latitude'), p_a.get('longitude')
                    lat_b, lon_b = p_b.get('latitude'), p_b.get('longitude')
                    if None in (lat_a, lon_a, lat_b, lon_b):
                        continue
                    R = 6371000
                    phi1 = _math.radians(lat_a)
                    phi2 = _math.radians(lat_b)
                    dphi = _math.radians(lat_b - lat_a)
                    dlambda = _math.radians(lon_b - lon_a)
                    a = _math.sin(dphi / 2) ** 2 + _math.cos(phi1) * _math.cos(phi2) * _math.sin(dlambda / 2) ** 2
                    c = 2 * _math.atan2(_math.sqrt(a), _math.sqrt(1 - a))
                    dist = R * c
                    dist_data.append({
                        "Foto A": os.path.basename(p_a['path']),
                        "Foto B": os.path.basename(p_b['path']),
                        "Distancia (m)": round(dist, 1),
                    })
            st.dataframe(dist_data, use_container_width=True)


# ---------------------------------------------------------------------------
# Función helper: ajustar línea de frontera (debe estar definida a nivel de módulo
# para poder usarse tanto en la pestaña de edición como en la de descarga)
# ---------------------------------------------------------------------------
def adjust_line(line_px, img_w, img_h, offset_px, angle_deg):
    if not line_px or len(line_px) < 2:
        return line_px
    x1, y1 = line_px[0]
    x2, y2 = line_px[-1]
    # Centro de la línea
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    # Vector dirección
    dx = x2 - x1
    dy = y2 - y1
    L = math.hypot(dx, dy) + 1e-9
    ux, uy = dx / L, dy / L
    # Perpendicular
    px, py = -uy, ux
    # Aplicar offset perpendicular
    cx += px * offset_px
    cy += py * offset_px
    # Aplicar rotación
    theta = math.radians(angle_deg)
    rx = ux * math.cos(theta) - uy * math.sin(theta)
    ry = ux * math.sin(theta) + uy * math.cos(theta)
    # Extender hasta los bordes de la imagen
    ts = []
    if abs(rx) > 1e-6:
        ts.append((0 - cx) / rx)
        ts.append((img_w - cx) / rx)
    if abs(ry) > 1e-6:
        ts.append((0 - cy) / ry)
        ts.append((img_h - cy) / ry)
    ts = [t for t in ts if not math.isinf(t)]
    if len(ts) >= 2:
        t1, t2 = min(ts), max(ts)
        return [(cx + t1 * rx, cy + t1 * ry), (cx + t2 * rx, cy + t2 * ry)]
    return line_px


# --- Pestañas principales ---
tab_map, tab_pairs, tab_preview, tab_download = st.tabs(
    ["🗺️ Mapa", "🔗 Pares", "✏️ Edición y Vista Previa", "⬇️ Descargar"]
)

# =====================================================================
# TAB 1: MAPA
# =====================================================================
with tab_map:
    st.subheader("Footprints y superposiciones")
    m = build_footprint_map(photos, pairs)
    st.components.v1.html(m._repr_html_(), height=600)

# =====================================================================
# TAB 2: LISTA DE PARES
# =====================================================================
with tab_pairs:
    st.subheader("Pares detectados")
    if not pairs:
        st.info("No se encontraron superposiciones con los criterios actuales.")
    else:
        for i, pair in enumerate(pairs):
            a_name = os.path.basename(pair['photo_a']['path'])
            b_name = os.path.basename(pair['photo_b']['path'])
            is_fallback = pair.get('intersection_polygon') is None
            area = pair['intersection_area_m2']
            dist = pair.get('fallback_gps_distance_m')
            if is_fallback:
                title = f"Par {i+1}: {a_name} ↔ {b_name} — Fallback GPS ({dist:.1f} m)"
            else:
                title = f"Par {i+1}: {a_name} ↔ {b_name} — {area:.1f} m²"
            with st.expander(title):
                if is_fallback:
                    st.warning("Este par fue detectado por distancia GPS (los footprints no intersectan). "
                               "La frontera puede ser menos precisa. Ajustala manualmente si es necesario.")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(pair['photo_a']['path'], caption=a_name, use_container_width=True)
                with col2:
                    st.image(pair['photo_b']['path'], caption=b_name, use_container_width=True)

# =====================================================================
# TAB 3: EDICIÓN Y VISTA PREVIA
# =====================================================================
with tab_preview:
    if not pairs:
        st.info("No hay pares para editar.")
    else:
        st.subheader("Ajuste de fronteras")
        st.markdown(
            "Para cada par superpuesto puedes ajustar la línea de separación "
            "usando los controles de **desplazamiento** (mover la línea) y "
            "**rotación** (inclinar la línea)."
        )

        # Cargar modelo YOLO si se necesita
        model = cached_yolo(model_name) if use_detector else None

        pair_idx = st.selectbox("Selecciona un par para editar", range(len(pairs)),
                                format_func=lambda i: f"Par {i+1}: {os.path.basename(pairs[i]['photo_a']['path'])} ↔ {os.path.basename(pairs[i]['photo_b']['path'])}")

        pair = pairs[pair_idx]
        photo_a = pair['photo_a']
        photo_b = pair['photo_b']

        # Detecciones (si el modelo está activo)
        dets_a, dets_b, matched = [], [], []
        if use_detector and model is not None:
            with st.spinner("Detectando aves en el par seleccionado..."):
                cls_filter = None if class_id < 0 else class_id
                dets_a = detect_birds(photo_a['path'], model, conf_threshold,
                                      tile_size=tile_size, class_id=cls_filter)
                dets_b = detect_birds(photo_b['path'], model, conf_threshold,
                                      tile_size=tile_size, class_id=cls_filter)
                matched = match_duplicates(dets_a, dets_b, photo_a, photo_b, max_match_distance)
            st.caption(f"Aves detectadas: {len(dets_a)} en A, {len(dets_b)} en B. Duplicados emparejados: {len(matched)}.")

        # Calcular frontera base
        boundary = compute_boundary(pair, matched, use_detector=use_detector)

        # Sliders de ajuste
        adj_key = f"pair_{pair_idx}"
        if adj_key not in st.session_state.pair_adjustments:
            st.session_state.pair_adjustments[adj_key] = {"offset": 0.0, "angle": 0.0}

        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            offset = st.slider(
                "Desplazamiento de frontera (px)",
                -200, 200,
                int(st.session_state.pair_adjustments[adj_key]["offset"]),
                1,
                key=f"offset_{pair_idx}"
            )
        with col_ctrl2:
            angle = st.slider(
                "Rotación de frontera (°)",
                -45, 45,
                int(st.session_state.pair_adjustments[adj_key]["angle"]),
                1,
                key=f"angle_{pair_idx}"
            )
        st.session_state.pair_adjustments[adj_key] = {"offset": offset, "angle": angle}

        img_w_a, img_h_a = photo_a['image_width'], photo_a['image_height']
        img_w_b, img_h_b = photo_b['image_width'], photo_b['image_height']

        line_a_adj = adjust_line(boundary['line_pixels_a'], img_w_a, img_h_a, offset, angle)
        line_b_adj = adjust_line(boundary['line_pixels_b'], img_w_b, img_h_b, offset, angle)

        # Vista previa con frontera dibujada
        col_a, col_b = st.columns(2)
        with col_a:
            img_a = Image.open(photo_a['path']).convert('RGB')
            draw_a = ImageDraw.Draw(img_a)
            if line_a_adj and len(line_a_adj) == 2:
                draw_a.line(line_a_adj, fill=(255, 0, 0), width=5)
            st.image(img_a, caption="Foto A con frontera (rojo)", use_container_width=True)

        with col_b:
            img_b = Image.open(photo_b['path']).convert('RGB')
            draw_b = ImageDraw.Draw(img_b)
            if line_b_adj and len(line_b_adj) == 2:
                draw_b.line(line_b_adj, fill=(255, 0, 0), width=5)
            st.image(img_b, caption="Foto B con frontera (rojo)", use_container_width=True)

        # Vista previa del resultado (máscara aplicada)
        st.markdown("---")
        st.subheader("Vista previa del resultado")

        # Puntos de referencia para la máscara: centroide de la otra foto proyectado
        cx_a, cy_a = photo_a['footprint']['centroid_utm']
        cx_b, cy_b = photo_b['footprint']['centroid_utm']
        H_a = boundary['homography_a']
        H_b = boundary['homography_b']
        ref_a = _utm_to_pixel(cx_b, cy_b, H_a, img_w_a, img_h_a)
        ref_b = _utm_to_pixel(cx_a, cy_a, H_b, img_w_b, img_h_b)

        mask_a = create_mask(img_w_a, img_h_a, line_a_adj, ref_a)
        mask_b = create_mask(img_w_b, img_h_b, line_b_adj, ref_b)

        img_a_preview = apply_white_mask(Image.open(photo_a['path']).convert('RGB'), mask_a)
        img_b_preview = apply_white_mask(Image.open(photo_b['path']).convert('RGB'), mask_b)

        col_pa, col_pb = st.columns(2)
        with col_pa:
            st.image(img_a_preview, caption="Foto A procesada (zona B en blanco)", use_container_width=True)
        with col_pb:
            st.image(img_b_preview, caption="Foto B procesada (zona A en blanco)", use_container_width=True)

# =====================================================================
# TAB 4: PROCESAMIENTO Y DESCARGA
# =====================================================================
with tab_download:
    st.subheader("Procesar todo el lote y descargar")
    st.markdown(
        "Haz clic en **Procesar todas las fotos** para aplicar las fronteras "
        "(con los ajustes manuales que hayas hecho) a TODO el lote y generar un ZIP."
    )

    if st.button("🚀 Procesar todas las fotos", type="primary"):
        with st.spinner("Procesando... esto puede tomar unos minutos."):
            model = cached_yolo(model_name) if use_detector else None

            # Inicializar máscaras acumuladas por foto
            masks = {}
            images = {}
            for p in photos:
                w = p.get('image_width', 1)
                h = p.get('image_height', 1)
                masks[p['path']] = np.zeros((h, w), dtype=bool)
                images[p['path']] = Image.open(p['path']).convert('RGB')

            progress = st.progress(0.0)
            for i, pair in enumerate(pairs):
                photo_a = pair['photo_a']
                photo_b = pair['photo_b']

                # Detecciones
                dets_a, dets_b, matched = [], [], []
                if use_detector and model is not None:
                    cls_filter = None if class_id < 0 else class_id
                    dets_a = detect_birds(photo_a['path'], model, conf_threshold,
                                          tile_size=tile_size, class_id=cls_filter)
                    dets_b = detect_birds(photo_b['path'], model, conf_threshold,
                                          tile_size=tile_size, class_id=cls_filter)
                    matched = match_duplicates(dets_a, dets_b, photo_a, photo_b, max_match_distance)

                boundary = compute_boundary(pair, matched, use_detector=use_detector)

                # Aplicar ajustes manuales si existen
                adj_key = f"pair_{i}"
                offset_px = st.session_state.pair_adjustments.get(adj_key, {}).get("offset", 0)
                angle_deg = st.session_state.pair_adjustments.get(adj_key, {}).get("angle", 0)

                img_w_a, img_h_a = photo_a['image_width'], photo_a['image_height']
                img_w_b, img_h_b = photo_b['image_width'], photo_b['image_height']

                line_a = adjust_line(boundary['line_pixels_a'], img_w_a, img_h_a, offset_px, angle_deg)
                line_b = adjust_line(boundary['line_pixels_b'], img_w_b, img_h_b, offset_px, angle_deg)

                # Referencias
                cx_a, cy_a = photo_a['footprint']['centroid_utm']
                cx_b, cy_b = photo_b['footprint']['centroid_utm']
                H_a = boundary['homography_a']
                H_b = boundary['homography_b']
                ref_a = _utm_to_pixel(cx_b, cy_b, H_a, img_w_a, img_h_a)
                ref_b = _utm_to_pixel(cx_a, cy_a, H_b, img_w_b, img_h_b)

                mask_a = create_mask(img_w_a, img_h_a, line_a, ref_a)
                mask_b = create_mask(img_w_b, img_h_b, line_b, ref_b)

                # Acumular (OR)
                masks[photo_a['path']] = np.logical_or(masks[photo_a['path']], mask_a)
                masks[photo_b['path']] = np.logical_or(masks[photo_b['path']], mask_b)

                progress.progress((i + 1) / len(pairs))

            # Aplicar máscaras finales
            processed = {}
            for p in photos:
                fname = os.path.basename(p['path'])
                img = apply_white_mask(images[p['path']], masks[p['path']])
                processed[fname] = img

            zip_bytes = build_download_zip(processed)
            st.session_state.processed_images = processed
            st.session_state.processed_ready = True
            st.success(f"¡Listo! {len(processed)} fotos procesadas.")

    if st.session_state.processed_ready:
        zip_bytes = build_download_zip(st.session_state.processed_images)
        st.download_button(
            label="⬇️ Descargar lote procesado (ZIP)",
            data=zip_bytes,
            file_name="fotos_aves_procesadas.zip",
            mime="application/zip",
        )

        # Mostrar miniaturas de algunas fotos procesadas
        with st.expander("Ver miniaturas del resultado"):
            cols = st.columns(4)
            for idx, (fname, img) in enumerate(st.session_state.processed_images.items()):
                with cols[idx % 4]:
                    st.image(img, caption=fname, use_container_width=True)
