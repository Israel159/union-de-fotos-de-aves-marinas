# Conteo de aves con deduplicación georreferenciada

App Streamlit para contar aves (pingüino, piquero, pelícano, zarcillo) en lotes de
fotos de dron con superposición, sin contar dos veces al mismo individuo.

## Cómo funciona

1. **Subir fotos**: lee lat/lon (EXIF), altitud y yaw (XMP de DJI) de cada foto.
2. **Detectar**: corre el detector YOLO foto a foto (cada ave se detecta donde mejor se ve).
3. **Emparejar**: propone parejas candidatas a duplicado (misma clase + centros GPS cercanos).
4. **Validar**: una persona ve los dos recortes de cada pareja y decide
   ✅ misma ave / ❌ distintas / ⏭ omitir. (Reemplaza al antiguo proceso manual de
   pintar de blanco: mismo efecto, sin editar imágenes y con registro de cada decisión.)
5. **Resultado**: conteo por clase, mapa y CSV descargable. Las uniones son
   transitivas: si A=B y B=C, las tres detecciones cuentan como un solo individuo.

## Instalación local

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp /ruta/a/tu/pinguino_piquero_pelicano.pt .      # raíz del proyecto
streamlit run app.py
```

## Despliegue en Streamlit Community Cloud

1. Crea una cuenta en GitHub (gratis) y otra en https://share.streamlit.io
   (entra con "Sign in with GitHub").
2. Sube este proyecto a un repo de GitHub. Opción A (web): en github.com pulsa
   "New repository", "Upload files" y arrastra TODO el contenido de esta carpeta
   (app.py, requirements.txt, la carpeta modules/ completa y
   `pinguino_piquero_pelicano.pt`). Opción B (terminal):
   ```bash
   git init && git add -A && git commit -m "app conteo aves"
   git branch -M main
   git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
   git push -u origin main
   ```
3. **Importante — el archivo de pesos**: GitHub rechaza archivos > 100 MB.
   Si tu `.pt` es más grande, súbelo a Google Drive/Dropbox con enlace de
   descarga directa y descomenta/configura `_W download_from_url` en
   `modules/detector.py`.
4. En https://share.streamlit.io pulsa **"New app"** → elige tu repo, rama
   `main`, archivo principal `app.py` → **"Deploy"**.
5. La primera compilación tarda varios minutos (instala PyTorch CPU ~600 MB).
   Cuando termina te da una URL pública tipo `https://tu-app.streamlit.app`.

### Límites del plan gratuito

- ~1 GB de RAM: torch + ultralytics caben, pero evita lotes de cientos de fotos
  pesadas en una sola sesión; procesa por lotes.
- La app "se duerme" tras inactividad y tarda ~1 min en despertar.
- **Sin almacenamiento persistente**: las fotos subidas y las decisiones viven
  solo en la sesión. Descarga el CSV del paso 5 antes de cerrar.
- No se necesitan secrets ni variables de entorno.

## Calibración del Phantom 3 Pro

Tu EXIF reporta focal 12.29 mm real / 24 mm equivalente, de donde la app deriva
un sensor de ~18.4 × 12.3 mm (modo "EXIF automático"). DJI, sin embargo, publica
un lente de 20 mm equiv. y FOV 94° (diagonal), que corresponde a ~84° × 62°
horiz./vert. y a una huella de ~1.80 × 1.20 × altitud. Los EXIF de DJI son
nominales, así que:

- Prueba ambos modos en la barra lateral ("EXIF automático" vs "FOV calibrado").
- Verifica con una pasada sobre un objeto de tamaño conocido (un bote, un auto,
  una persona) a altitud conocida y compara el tamaño en píxeles con el esperado.
- La validación visual del paso 4 te protege de errores de huella: si el
  emparejamiento GPS propone parejas incorrectas, recházalas con ❌.
