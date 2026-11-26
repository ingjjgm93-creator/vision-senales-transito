# ============================================================================
# Contenedor del pipeline de detección de señales de tránsito (Faster R-CNN)
# Visión Inteligente · Unidad IV · Actividad 4.3
# ============================================================================
# Imagen base oficial de Python (ligera). Se puede cambiar con --build-arg.
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="vision-senales-transito" \
      org.opencontainers.image.description="Pipeline Faster R-CNN para detectar señales de tránsito" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.authors="José de Jesús Guerrero Martínez"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1) Dependencias primero: Docker reutiliza esta capa si solo cambia el código.
#    El índice CPU de PyTorch evita descargar ~3 GB de librerías CUDA.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --index-url ${TORCH_INDEX} --extra-index-url https://pypi.org/simple -r requirements.txt

# 2) Código, modelo entrenado y datos de prueba
COPY src/ src/
COPY models/ models/
COPY data/ data/
COPY tests/ tests/

# 3) Usuario sin privilegios y carpeta de resultados
RUN useradd --create-home appuser && mkdir -p /app/output && chown -R appuser /app
USER appuser
VOLUME ["/app/output"]

# El contenedor es el pipeline de la Actividad 4.1 con el modelo ya indicado.
# Por defecto procesa las 15 imágenes de validación en lotes de 4.
ENTRYPOINT ["python", "src/pipeline_inferencia.py", "--modelo", "models/modelo_senales_frcnn.pth"]
CMD ["--entrada", "data/dataset/images/val", "--salida", "output", "--lote", "4"]
