# Union de Fotos de Aves Marinas

Aplicación Streamlit para eliminar zonas superpuestas entre fotos de dron georreferenciadas, preparándolas para algoritmos de reconocimiento de imágenes (por ejemplo, conteo de aves marinas con YOLO).

## Problema que resuelve

Cuando un dron fotografía un área con **overlap** (superposición entre fotos consecutivas), el mismo ave puede aparecer en dos o más imágenes. Si se corre un detector de objetos directamente, esa ave se cuenta **dos veces** (o más). Esta app:

1. Lee la georreferencia (GPS + altitud + orientación del gimbal) del EXIF/XMP de cada foto.
2. Calcula el **footprint** (área fotografiada en el suelo) de cada imagen.
3. Detecta automáticamente qué pares de fotos se superponen.
4. Dibuja una **frontera de separación** entre cada par:
   - **Base**: línea media geométrica entre los centroides de las fotos (solo GPS).
   - **Refinamiento**: si se activa, usa un detector genérico de aves (YOLOv8) para encontrar aves duplicadas en la zona de overlap y ajusta la frontera para pasar **entre** ellas, evitando cortar ningún ave por la mitad.
5. Permite **editar manualmente** cada frontera (desplazamiento y rotación).
6. Pinta de **blanco** la zona superpuesta que corresponde a la otra foto.
7. Genera un **ZIP** con todas las fotos procesadas listas para el detector de especies.

## Estructura del repositorio

```
union-de-fotos-de-aves-marinas/
├── app.py                      # Interfaz Streamlit
├── requirements.txt            # Dependencias Python
├── packages.txt                # Dependencias del sistema (Streamlit Cloud)
├── README.md                   # Este archivo
├── .gitignore
├── modules/
│   ├── __init__.py
│   ├── exif_reader.py          # Lectura EXIF/XMP (DJI)
│   ├── footprint.py            # Cálculo de footprints georreferenciados
│   ├── overlap.py              # Detección de pares superpuestos
│   ├── detector.py             # Detección genérica de aves con YOLOv8
│   ├── boundary.py             # Cálculo de frontera híbrida
│   ├── processor.py            # Aplicación de máscaras y generación de ZIP
│   └── utils.py                # Visualización (mapas, etc.)
├── models/                     # (vacío; YOLO se descarga automáticamente)
└── assets/                     # (vacío)
```

## Despliegue en Streamlit Cloud

1. **Crea un repositorio en GitHub** con el contenido de esta carpeta.
2. Ve a [share.streamlit.io](https://share.streamlit.io) y conecta tu repositorio.
3. El archivo de entrada es `app.py`.
4. La primera vez que se ejecute, descargará automáticamente el modelo `yolov8n.pt` (~6 MB).

## Uso local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notas importantes

- **Detector genérico vs. detector de especies**: esta app usa `yolov8n.pt` (clase `bird` del dataset COCO) **solo** para calcular la frontera entre fotos. Tu modelo `pinguino_pelicano_piquero.pt` (o similar) debe usarse en un paso posterior, sobre las fotos ya limpias.
- **Si el detector falla**: la app nunca se queda sin frontera. Si no detecta aves o no hay duplicados, automáticamente usa la línea media geométrica basada en GPS.
- **Edición manual**: en la pestaña "Edición y Vista Previa" puedes ajustar cada frontera con sliders de desplazamiento y rotación.
- **Altitud**: la app intenta usar `RelativeAltitude` del XMP de DJI. Si no existe, usa `GPSAltitude` como fallback.

## Fotos compatibles

- JPEG con metadatos EXIF estándar (GPS, focal length).
- XMP de DJI (Phantom, Mavic, etc.) para orientación del gimbal y altitud relativa.
- Probado con DJI Phantom 3 Professional.

## Licencia

MIT
