#!/usr/bin/env bash
# =============================================================================
#  Actividad 4.3 · Visión Inteligente · construir_y_ejecutar.sh (Linux / macOS)
#  Construye la imagen, corre las pruebas y todos los modos, y deja en
#  output/resumen_para_reporte.txt los datos de las tablas 10 y 11 del reporte.
#
#  Uso:  bash scripts/construir_y_ejecutar.sh
# =============================================================================
set -u
cd "$(dirname "$0")/.."
RAIZ="$(pwd)"
IMAGEN="vision-senales-transito:1.0"
LOGS="$RAIZ/output/logs"
mkdir -p "$LOGS" ejemplos data/video
chmod 777 output 2>/dev/null || true

titulo() { printf '\n%s\n  %s\n%s\n' "$(printf '=%.0s' {1..76})" "$1" "$(printf '=%.0s' {1..76})"; }
ejecutar() { local log="$1"; shift; echo "\$ $*"; "$@" 2>&1 | tee "$LOGS/$log"; return "${PIPESTATUS[0]}"; }

titulo "0. Revisión de requisitos"
docker info >/dev/null 2>&1 || { echo "ERROR: Docker no responde."; exit 1; }
[ -f models/modelo_senales_frcnn.pth ] || { echo "ERROR: falta models/modelo_senales_frcnn.pth"; exit 1; }
[ -d data/dataset/images/val ] || { echo "ERROR: falta data/dataset (images/, labels/, data.yaml)"; exit 1; }

W="https://upload.wikimedia.org/wikipedia/commons/thumb"
descargar() { [ -f "ejemplos/$1" ] || { curl -fsSL -A "VisionInteligenteTecNM/1.0 (actividad academica)" -o "ejemplos/$1" "$2" && echo "Descargada: ejemplos/$1"; sleep 1; }; }
descargar alto.jpg "$W/7/71/ALTO_sign.jpg/500px-ALTO_sign.jpg"
descargar ceda_el_paso.png "$W/b/b9/Mexico_traffic_signal_sr7.svg/500px-Mexico_traffic_signal_sr7.svg.png"
descargar no_estacionar.jpg "$W/1/1f/Se%C3%B1al_de_no_estacionarse.jpg/500px-Se%C3%B1al_de_no_estacionarse.jpg"
descargar velocidad.png "$W/2/2d/Mexico_road_sign_SR-09.svg/500px-Mexico_road_sign_SR-09.svg.png"

titulo "1. Datos del equipo"
SO="$(uname -sr)"; [ -f /etc/os-release ] && SO="$(. /etc/os-release; echo "$PRETTY_NAME") ($(uname -r))"
if command -v sysctl >/dev/null && sysctl -n machdep.cpu.brand_string >/dev/null 2>&1; then
  CPU="$(sysctl -n machdep.cpu.brand_string)"; NUC="$(sysctl -n hw.ncpu) hilos"; RAM="$(( $(sysctl -n hw.memsize) / 1073741824 ))"
else
  CPU="$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"; NUC="$(nproc) hilos"; RAM="$(awk '/MemTotal/{printf "%.1f", $2/1048576}' /proc/meminfo)"
fi
GPU="$(command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo 'sin GPU NVIDIA')"
DOCKER="$(docker version --format '{{.Server.Platform.Name}} (Engine {{.Server.Version}})' 2>/dev/null)"
echo "SO: $SO | CPU: $CPU ($NUC) | RAM: $RAM GB | GPU: $GPU | Docker: $DOCKER"

titulo "2. Construcción de la imagen"
INICIO=$(date +%s)
ejecutar 01_build.txt docker build -t "$IMAGEN" . || { echo "ERROR: falló docker build"; exit 1; }
BUILD_S=$(( $(date +%s) - INICIO ))
if [ ! -f data/video/camara_simulada.mp4 ]; then
  ejecutar 01b_video_demo.txt docker run --rm --user root --entrypoint python -v "$RAIZ/data:/app/data" "$IMAGEN" src/crear_video_demo.py
  ejecutar 01c_rebuild.txt docker build -t "$IMAGEN" .
fi

titulo "3. Imagen generada"
ejecutar 03_images.txt docker images vision-senales-transito
TAM=$(docker image inspect "$IMAGEN" --format '{{.Size}}' | awk '{printf "%.2f", $1/1e9}')
cat > "$LOGS/info_equipo.json" <<EOF
{"so": "$SO", "cpu": "$CPU", "nucleos": "$NUC", "ram_gb": "$RAM", "gpu": "$GPU", "docker": "$DOCKER", "build_s": $BUILD_S, "tam_gb": $TAM}
EOF

VOL=(-v "$RAIZ/output:/app/output")
EJ=(-v "$RAIZ/ejemplos:/app/ejemplos:ro")
titulo "4. Pruebas de humo";        ejecutar 02_pruebas.txt docker run --rm --entrypoint python "$IMAGEN" tests/test_pipeline.py
titulo "5. Modo por defecto";       ejecutar 04_pipeline.txt docker run --rm "${VOL[@]}" "$IMAGEN"
titulo "6. Evaluación";             ejecutar 05_evaluacion.txt docker run --rm "${VOL[@]}" --entrypoint python "$IMAGEN" src/evaluar.py --salida output/evaluacion
titulo "7. Video";                  ejecutar 06_video.txt docker run --rm "${VOL[@]}" "$IMAGEN" --entrada data/video/camara_simulada.mp4 --salida output/video
titulo "8. Internet, umbral 0.6";   ejecutar 07_externas_06.txt docker run --rm "${EJ[@]}" "${VOL[@]}" "$IMAGEN" --entrada ejemplos --salida output/externas
titulo "9. Internet, umbral 0.3";   ejecutar 08_externas_03.txt docker run --rm "${EJ[@]}" "${VOL[@]}" "$IMAGEN" --entrada ejemplos --conf 0.3 --salida output/externas_03

titulo "10. Datos para el reporte"
docker run --rm "${VOL[@]}" -v "$RAIZ/scripts:/app/scripts:ro" --entrypoint python "$IMAGEN" scripts/resumen_reporte.py --salida output
echo; echo "Listo. Revisa output/resumen_para_reporte.txt"
