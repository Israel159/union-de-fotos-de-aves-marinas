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
├── debug.py                    # Diagnóstico de metadatos y superposiciones
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
- **Altitud**: la app usa `RelativeAltitude` cuando es positiva. Si falta o es negativa, utiliza `AbsoluteAltitude`/`GPSAltitude` como aproximación y lo indica en el modo debug. La altitud absoluta no equivale necesariamente a la altura sobre el suelo; si el terreno está muy por encima o por debajo del nivel del mar, el tamaño del footprint será aproximado.
- **Cámara inclinada**: cuando el gimbal no está en nadir, las esquinas se proyectan mediante rayos sobre un plano de suelo horizontal. El GPS del dron deja de tratarse como centro de la foto. Esto no sustituye un modelo de terreno ni una orientación fotogramétrica completa, especialmente en acantilados o con mucha pendiente.
- **Fallback GPS**: la casilla de la barra lateral controla si se buscan fotos cercanas cuando no existe ninguna intersección real. Un par marcado como fallback tiene 0 m² de intersección calculada y debe revisarse manualmente.

## Diagnóstico de un lote

```bash
python debug.py /ruta/a/carpeta/con/fotos
```

El script genera `debug_report.json` y `debug_map.html` dentro de esa carpeta. El reporte incluye la versión de Shapely, altitudes original y utilizada, orientación, tipo de proyección, footprints y distinción entre intersecciones reales y pares de fallback. La detección de pares es compatible tanto con Shapely 1 (el índice devuelve geometrías) como con Shapely 2 (devuelve índices numéricos).

## Fotos compatibles

- JPEG y TIFF con metadatos EXIF estándar (GPS, focal length).
- XMP de DJI (Phantom, Mavic, etc.) para orientación del gimbal y altitud relativa.
- Probado con DJI Phantom 3 Professional.

## Modelo YOLO configurable

Para usar un modelo diferente (por ejemplo `yolov8l.pt`):
1. Coloca el archivo `.pt` dentro de la carpeta `models/`.
2. En el panel de configuración (sidebar), cambia el nombre del modelo.
3. Si el archivo no existe localmente, se descargará automáticamente la primera vez.

## Licencia

MIT
