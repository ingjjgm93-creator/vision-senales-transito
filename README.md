# Detección de señales de tránsito con Faster R-CNN en Docker

Actividad 4.3: integración y empaquetado del proyecto de visión
Maestría en Inteligencia Artificial · TecNM Virtual · Visión Inteligente, Unidad IV
Alumno: José de Jesús Guerrero Martínez (26ZPM004) · Docente: Mtro. Luis Alberto Espejo Ponce

El contenedor recibe una imagen (de internet, de una cámara o del dataset) y responde con la clase de cada señal, su porcentaje de confianza y su caja. Solo hace inferencia; no entrena.

| Pieza | Origen | Archivo |
|---|---|---|
| Dataset propio: 75 imágenes de 640×640 px (60 train, 15 val), 154 señales, etiquetas YOLO | Actividad 1.4 | `data/dataset/` |
| Detector Faster R-CNN MobileNetV3-Large-320 FPN, checkpoint con metadatos (76.1 MB) | Actividad 4.1 | `models/modelo_senales_frcnn.pth` |
| Pipeline de inferencia: entrada, preprocesamiento, inferencia por lotes, NMS, salidas y log | Actividad 4.1 | `src/pipeline_inferencia.py` |

Clases: `alto`, `ceda_el_paso`, `no_estacionar`, `velocidad`. Umbral de operación: confianza 0.6, NMS 0.5.

## Estructura

```
vision-senales-transito/
├── Dockerfile                    receta de la imagen
├── docker-compose.yml            ejecución con un solo comando
├── requirements.txt              dependencias con versión fija
├── README.md
├── .dockerignore / .gitignore
├── src/
│   ├── pipeline_inferencia.py    pipeline de la Actividad 4.1
│   ├── evaluar.py                métricas contra las etiquetas YOLO
│   └── crear_video_demo.py       video de cámara simulada
├── models/modelo_senales_frcnn.pth
├── data/
│   ├── dataset/                  images/, labels/ y data.yaml (Unidad 1)
│   └── video/camara_simulada.mp4 60 cuadros a 15 FPS
├── ejemplos/                     imágenes de internet para probar
├── tests/test_pipeline.py        pruebas de humo
├── scripts/                      construir_y_ejecutar.ps1 (Windows) y .sh (Linux/macOS)
└── output/                       resultados (volumen; no se sube a GitHub)
```

## Dataset y archivos incluidos

El checkpoint `models/modelo_senales_frcnn.pth` y el dataset completo de la Unidad I están incluidos en este paquete. El dataset conserva `images/train`, `images/val`, `labels/train`, `labels/val`, `exports/coco`, `exports/pascal_voc`, `analysis` y `data.yaml`. La ruta de `data.yaml` se corrigió a `path: .` para evitar dependencias de rutas absolutas.

El conjunto contiene 75 imágenes (60 de entrenamiento y 15 de validación), 154 anotaciones y cuatro clases: `alto`, `ceda_el_paso`, `no_estacionar` y `velocidad`. Consulta `data/dataset/README_VALIDACION.md` para el resumen de integridad.

Los scripts de ejecución descargan las cuatro imágenes externas de Wikimedia Commons a `ejemplos/` y generan el video simulado a partir de las imágenes de validación. Si la descarga automática falla, revisa `ejemplos/` y descarga manualmente los recursos indicados en `ejemplos/FUENTES.md`.

La carpeta `output/` se genera durante la ejecución y no debe subirse al repositorio.

## Uso rápido en Windows con Docker Desktop

1. Abre Docker Desktop y espera a que diga *Engine running*.
2. Clona el repositorio y abre PowerShell en su carpeta.
3. Asegúrate de haber agregado `dataset.zip` a la raíz o de haber preparado `data/dataset/`. El script automatizado también descarga las imágenes externas. Después construye la imagen:
   ```powershell
   docker build -t vision-senales-transito:1.0 .
   ```
4. Ejecuta el modo por defecto (las 15 imágenes de validación en lotes de 4):
   ```powershell
   docker run --rm -v "${PWD}\output:/app/output" vision-senales-transito:1.0
   ```
5. Revisa `output\resultados.json` y las imágenes `*_det.jpg`.

Todo en un paso (construye, prueba, ejecuta cada modo y resume los resultados en `output\resumen_para_reporte.txt`):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\construir_y_ejecutar.ps1
```

En Linux o macOS: `bash scripts/construir_y_ejecutar.sh`.

## Probar con una imagen nueva

Copia la foto (descargada de internet o tomada con el celular) a la carpeta `ejemplos/` y monta esa carpeta:

```powershell
docker run --rm -v "${PWD}\ejemplos:/app/ejemplos:ro" -v "${PWD}\output:/app/output" `
  vision-senales-transito:1.0 --entrada ejemplos --salida output/externas
```

Salida por imagen en `output/externas/resultados.json`:

```json
{"clase": "alto", "clase_id": 1, "confianza": 0.98, "caja": [112.4, 80.0, 391.7, 362.5]}
```

`confianza` va de 0 a 1 (0.98 = 98 %) y `caja` es `[x1, y1, x2, y2]` en píxeles de la imagen original. La imagen anotada queda junto al JSON.

## Otros modos

```powershell
# Métricas contra las etiquetas YOLO (P, R, F1, AP@0.5 por clase, mAP@0.5, IoU medio)
docker run --rm -v "${PWD}\output:/app/output" --entrypoint python vision-senales-transito:1.0 src/evaluar.py --salida output/evaluacion

# Video (tiempo real)
docker run --rm -v "${PWD}\output:/app/output" vision-senales-transito:1.0 --entrada data/video/camara_simulada.mp4 --salida output/video

# Umbral de confianza distinto
docker run --rm -v "${PWD}\ejemplos:/app/ejemplos:ro" -v "${PWD}\output:/app/output" vision-senales-transito:1.0 --entrada ejemplos --conf 0.3 --salida output/externas_03

# Pruebas de humo
docker run --rm --entrypoint python vision-senales-transito:1.0 tests/test_pipeline.py

# Docker Compose
docker compose up --build
docker compose --profile extra run evaluacion
docker compose --profile extra run video
```

## Notas

- La imagen usa CPU. El Dockerfile prioriza el índice oficial de PyTorch para CPU (`--index-url`) y utiliza PyPI como índice complementario (`--extra-index-url`), evitando resolver primero PyTorch desde PyPI.
- La imagen no se publica en ningún registro: se construye en cada equipo a partir del Dockerfile.
- El modelo pesa 76.1 MB, por debajo del límite de 100 MB por archivo de GitHub, y se versiona directo (GitHub muestra una advertencia por pasar de 50 MB).
- El contenedor no accede a la cámara web en Windows ni en macOS; el modo de cámara se prueba con el video simulado.
- Las métricas perfectas se miden con imágenes del mismo dataset sintético. Con fotografías reales el desempeño baja; ver el reporte, sección 7.6.
