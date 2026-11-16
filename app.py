"""App de conteo de aves con deduplicación georreferenciada.

Flujo:
  1. SUBIR fotos (lee EXIF GPS + XMP de DJI; soporta JPG y TIFF)
  2. DETECTAR aves en cada foto (YOLO)
  3. EMPAREJAR detecciones candidatas a duplicado (misma clase + cercanía GPS)
  4. VALIDAR cada pareja visualmente (misma ave / diferentes)
  5. RESULTADO: individuos únicos, conteo por clase y mapa

Cambios respecto a la versión anterior:
- Acepta .tif/.tiff (y .png, para dar un mensaje claro si el compresor web
  eliminó los metadatos).
- Fotos con GPS pero sin altitud ya NO se omiten: se les aplica la
  "Altitud por defecto (m)" de la barra lateral y se avisa en pantalla.
"""
import io
import math
import pandas as pd
import streamlit as st
from PIL import Image

from modules.exif_utils import extract_metadata
from modules.geo import (bbox_center_and_ground_size, footprint_size_m,
                         footprint_from_fov, offset_to_latlon,
                         pixel_to_ground_offset)
from modules.detector import get_detector
from modules.matching import build_candidates, cluster_detections

st.set_page_config(page_title="Conteo de aves — deduplicación", layout="wide")

CLASES = ["pinguino", "piquero", "pelicano", "zarcillo"]

# ----------------------------- estado global -------------------------------
ss = st.session_state
ss.setdefault("photos", {})       # nombre -> {pil, meta, g_w, g_h}
ss.setdefault("detections", [])   # lista plana de detecciones georreferenciadas
ss.setdefault("candidates", [])
ss.setdefault("decisions", {})    # id_candidato -> True/False
ss.setdefault("val_idx", 0)

# ----------------------------- utilidades ----------------------------------
def crop_with_margin(pil_img, bbox, margin=0.4):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    mx, my = int(w * margin), int(h * margin)
    return pil_img.crop((max(0, x1 - mx), max(0, y1 - my),
                         min(pil_img.width, x2 + mx), min(pil_img.height, y2 + my)))

def pending_candidates():
    return [c for c in ss.candidates if c["id"] not in ss.decisions]

# ----------------------------- barra lateral -------------------------------
st.sidebar.title("Pipeline")
step = st.sidebar.radio("Paso", ["1. Subir fotos", "2. Detectar", "3. Emparejar",
                                 "4. Validar uniones", "5. Resultado"])

with st.sidebar.expander("Ajustes del detector y GPS"):
    conf_min = st.slider("Confianza mínima del detector (YOLO)", 0.05, 0.95, 0.40,
                         help="Solo se aceptan detecciones con confianza >= este valor. "
                              "Si te salen muchas falsas positivas, súbelo; si se te "
                              "escapan aves, bájalo.")
    umbral_m = st.number_input("Distancia máx. entre candidatos (m) — 0 = automático",
                               min_value=0.0, value=0.0, step=0.5,
                               help="Dos detecciones de la misma clase a menos de esta "
                                    "distancia en el suelo se proponen para unir. "
                                    "Empieza con 0 (automático) y ajústalo tras ver resultados.")
    dist_factor = st.slider("Factor de umbral automático", 0.5, 3.0, 1.2)
    alt_default = st.number_input(
        "Altitud por defecto (m)", min_value=1.0, value=50.0, step=5.0,
        help="Se aplica a las fotos que tienen GPS pero perdieron la altitud "
             "(p. ej. las pasadas por un compresor web que borra el XMP). "
             "Idealmente súbelo a la altitud de vuelo de tu dron (la relativa "
             "sobre el terreno).")

    st.caption("Huella en el suelo (elige el modo):")
    modo_huella = st.radio("Modo de huella",
                           ["EXIF automático",
                            "FOV calibrado (recomendado P3P)"],
                           help="EXIF automático: deriva el sensor de FocalLength y "
                                "FocalLengthIn35mmFilm. FOV calibrado: usa el campo "
                                "visual de la cámara (más fiable si el EXIF de DJI "
                                "es nominal). Phantom 3 Pro oficial: 84° x 62°.")
    if modo_huella.startswith("FOV"):
        fov_h = st.number_input("FOV horizontal (°)", 10.0, 170.0, 84.0, format="%.1f")
        fov_v = st.number_input("FOV vertical (°)", 10.0, 170.0, 62.0, format="%.1f")
    else:
        fov_h = fov_v = None
        st.caption("Cámara (fallback si el EXIF no trae sensor):")
        def_sensor_w = st.number_input("Ancho sensor (mm)", 1.0, 50.0, 18.4, format="%.1f")
        def_sensor_h = st.number_input("Alto sensor (mm)", 1.0, 50.0, 12.3, format="%.1f")
        def_focal = st.number_input("Focal fallback (mm)", 1.0, 50.0, 12.29, format="%.2f")

# ----------------------------- paso 1: subir -------------------------------
if step == "1. Subir fotos":
    st.header("1. Subir lote de fotos")
    files = st.file_uploader("Fotos del dron (JPG/TIFF con EXIF/GPS)",
                             type=["jpg", "jpeg", "tif", "tiff", "png"],
                             accept_multiple_files=True)
    if files:
        for f in files:
            if f.name in ss.photos:
                continue
            raw = f.read()
            path = f"/tmp/{f.name}"
            with open(path, "wb") as out:
                out.write(raw)
            try:
                meta = extract_metadata(path)
            except ValueError as e:
                st.warning(f"{f.name}: {e}")
                continue
            pil = Image.open(io.BytesIO(raw))
            if meta["alt_m"] is None:
                meta["alt_m"] = float(alt_default)
                meta["alt_default"] = True
                st.info(f"{f.name}: sin altitud en los metadatos — se usa la "
                        f"altitud por defecto ({alt_default:.0f} m). "
                        "Si el dron volaba a otra altura, cámbiala en la barra lateral.")
            else:
                meta["alt_default"] = False
            if modo_huella.startswith("FOV"):
                g_w = footprint_from_fov(meta["alt_m"], fov_h)
                g_h = footprint_from_fov(meta["alt_m"], fov_v)
                st.caption(f"{f.name}: huella por FOV {fov_h}°x{fov_v}° "
                           f"= {g_w:.1f} x {g_h:.1f} m a {meta['alt_m']:.1f} m de altitud.")
            else:
                sensor_w = meta.get("sensor_w_mm") or def_sensor_w
                sensor_h = meta.get("sensor_h_mm") or def_sensor_h
                focal = meta.get("focal_mm") or def_focal
                if meta.get("sensor_w_mm") is None:
                    st.info(f"{f.name}: sensor no legible en EXIF, usando valores "
                            f"manuales ({def_sensor_w}×{def_sensor_h} mm, f={def_focal} mm).")
                g_w = footprint_size_m(meta["alt_m"], focal, sensor_w)
                g_h = footprint_size_m(meta["alt_m"], focal, sensor_h)
            ss.photos[f.name] = {"pil": pil, "meta": meta, "g_w": g_w, "g_h": g_h}
        st.success(f"{len(ss.photos)} fotos cargadas con GPS.")
        st.dataframe(pd.DataFrame([{**v["meta"], "foto": k}
                                   for k, v in ss.photos.items()]))
        n_def = sum(1 for v in ss.photos.values() if v["meta"].get("alt_default"))
        if n_def:
            st.warning(f"{n_def} foto(s) usan la altitud por defecto. El tamaño de "
                       "la huella (y por tanto las coordenadas de cada ave) será "
                       "aproximado: corrige la altitud real de vuelo y vuelve a "
                       "subir el lote para mayor precisión.")
        st.caption("Verifica que alt_m sea razonable (la relativa del XMP es la "
                   "mejor) y que yaw_deg exista. Si el EXIF de DJI es nominal, "
                   "prueba el modo de huella por FOV en la barra lateral.")

# ----------------------------- paso 2: detectar ----------------------------
elif step == "2. Detectar":
    st.header("2. Ejecutar detector por foto")
    if not ss.photos:
        st.info("Primero sube fotos en el paso 1.")
    elif st.button("Detectar aves", type="primary"):
        try:
            detector = get_detector()
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        ss.detections = []
        bar = st.progress(0.0)
        for n, (name, p) in enumerate(ss.photos.items()):
            for det in detector.predict(p["pil"]):
                if det["conf"] < conf_min:
                    continue
                cx, cy, size_m = bbox_center_and_ground_size(
                    det, p["meta"]["img_w"], p["meta"]["img_h"], p["g_w"], p["g_h"])
                dx, dy = pixel_to_ground_offset(
                    cx, cy, p["meta"]["img_w"], p["meta"]["img_h"],
                    p["g_w"], p["g_h"], p["meta"]["yaw_deg"] or 0.0)
                lat, lon = offset_to_latlon(p["meta"]["lat"], p["meta"]["lon"], dx, dy)
                ss.detections.append({**det, "img": name, "lat": lat, "lon": lon,
                                      "size_m": size_m})
            bar.progress((n + 1) / len(ss.photos))
        st.success(f"{len(ss.detections)} detecciones en {len(ss.photos)} fotos.")
    if ss.detections:
        st.dataframe(pd.DataFrame(ss.detections))

# ----------------------------- paso 3: emparejar ---------------------------
elif step == "3. Emparejar":
    st.header("3. Buscar candidatas a duplicado")
    if not ss.detections:
        st.info("Primero ejecuta el detector (paso 2).")
    else:
        ss.candidates = build_candidates(ss.detections, max_dist_m=umbral_m or None,
                                         dist_factor=dist_factor)
        st.metric("Parejas candidatas a revisar", len(ss.candidates))
        st.caption("Parejas con la misma clase cuyo centro GPS está dentro del umbral.")
        if ss.candidates:
            st.dataframe(pd.DataFrame([
                {"id": c["id"], "clase": ss.detections[c["a"]]["class"],
                 "foto A": ss.detections[c["a"]]["img"],
                 "foto B": ss.detections[c["b"]]["img"],
                 "dist_m": c["dist_m"], "umbral_m": c["umbral_m"]}
                for c in ss.candidates]))

# ----------------------------- paso 4: validar -----------------------------
elif step == "4. Validar uniones":
    st.header("4. Validación visual de cada pareja")
    pend = pending_candidates()
    if not ss.candidates:
        st.info("Primero genera candidatas (paso 3).")
    elif not pend:
        st.success(f"Todas las parejas validadas ({len(ss.decisions)} decisiones). "
                   "Ve al paso 5.")
        if st.button("Revisar de nuevo"):
            ss.decisions = {}
            st.rerun()
    else:
        c = pend[0]
        da, db = ss.detections[c["a"]], ss.detections[c["b"]]
        col1, col2, col3 = st.columns([2, 2, 1.2])
        with col1:
            st.subheader("Detección A")
            st.image(crop_with_margin(ss.photos[da["img"]]["pil"], da["bbox"]),
                     caption=f'{da["img"]} — {da["class"]} ({da["conf"]:.2f})')
        with col2:
            st.subheader("Detección B")
            st.image(crop_with_margin(ss.photos[db["img"]]["pil"], db["bbox"]),
                     caption=f'{db["img"]} — {db["class"]} ({db["conf"]:.2f})')
        with col3:
            st.metric("Pendientes", len(pend))
            st.write(f"**Clase:** {da['class']}")
            st.write(f"**Distancia GPS:** {c['dist_m']} m (umbral {c['umbral_m']} m)")
            st.write(f"**Coord. A:** {da['lat']:.6f}, {da['lon']:.6f}")
            st.write(f"**Coord. B:** {db['lat']:.6f}, {db['lon']:.6f}")
            st.divider()
            if st.button("✅ Misma ave (unir)", type="primary", use_container_width=True):
                ss.decisions[c["id"]] = True
                st.rerun()
            if st.button("❌ Son aves distintas", use_container_width=True):
                ss.decisions[c["id"]] = False
                st.rerun()
            if st.button("⏭ Omitir por ahora", use_container_width=True):
                ss.candidates.append(ss.candidates.pop(ss.candidates.index(c)))
                st.rerun()

# ----------------------------- paso 5: resultado ---------------------------
else:
    st.header("5. Conteo final de individuos")
    if not ss.detections:
        st.info("Pipeline vacío. Sube fotos y ejecuta los pasos anteriores.")
    else:
        grupos = cluster_detections(ss.detections, ss.candidates, ss.decisions)
        individuos = []
        for g in grupos:
            best = max(g, key=lambda i: ss.detections[i]["conf"])
            d = ss.detections[best]
            individuos.append({"clase": d["class"], "lat": d["lat"], "lon": d["lon"],
                               "conf": d["conf"], "foto": d["img"],
                               "detecciones_unidas": len(g)})
        df = pd.DataFrame(individuos)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Conteo por clase")
            st.table(df["clase"].value_counts().reindex(CLASES, fill_value=0)
                     .rename_axis("clase").rename("individuos"))
        with c2:
            st.subheader("Mapa de individuos")
            st.map(df.rename(columns={"lat": "latitude", "lon": "longitude"}))
        st.dataframe(df)
        st.download_button("Descargar CSV", df.to_csv(index=False).encode(),
                           "conteo_aves.csv", "text/csv")
        st.caption(f"{len(ss.detections)} detecciones → {len(individuos)} individuos "
                   f"únicos (factor de compresión "
                   f"{len(ss.detections)/max(len(individuos),1):.2f}x).")
