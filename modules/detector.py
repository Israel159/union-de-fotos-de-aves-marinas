"""Interfaz del detector de aves (YOLO/ultralytics).

Coloca el archivo de pesos 'pinguino_piquero_pelicano.pt' en la RAÍZ del
repositorio (misma carpeta que app.py).

Si tus pesos pesan más de 100 MB (límite de GitHub), súbelos a Google Drive
o Dropbox, crea un enlace de descarga directa y descomenta el bloque
_W download_from_url de abajo.
"""
import os
import streamlit as st

WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "pinguino_piquero_pelicano.pt")

# _W download_from_url = "https://...enlace-directo.../pinguino_piquero_pelicano.pt"


class YoloDetector:
    def __init__(self, weights_path=WEIGHTS):
        from ultralytics import YOLO
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"No se encontró el archivo de pesos: {weights_path}\n"
                "Copia 'pinguino_piquero_pelicano.pt' a la raíz del repo, "
                "o configura la descarga desde URL en modules/detector.py.")
        self.model = YOLO(weights_path)

    def predict(self, pil_img):
        """Devuelve lista de {'bbox':[x1,y1,x2,y2], 'class':str, 'conf':float}."""
        res = self.model(pil_img, verbose=False)[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls = int(b.cls[0])
            name = str(res.names[cls]).strip().lower()
            out.append({"bbox": [x1, y1, x2, y2], "class": name,
                        "conf": float(b.conf[0])})
        return out


@st.cache_resource
def _detector_cached():
    return YoloDetector()


def get_detector():
    return _detector_cached()
