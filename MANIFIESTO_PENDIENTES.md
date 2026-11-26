# Estado del paquete de la Actividad 4.3

## Incluido en esta versión
- README.md
- requirements.txt
- src/ (pipeline, evaluación y video demo)
- Dockerfile y docker-compose.yml
- tests/
- scripts/ (automatización de Windows y Linux/macOS)
- models/modelo_senales_frcnn.pth (76,081,345 bytes)
  SHA-256: adcd33791ce4bf1bba91eb1457385e84793cf774451c7b2f741e0f3d0de01b33
- data/dataset/ (dataset completo de la Actividad 1.4)
  75 imágenes, 75 etiquetas YOLO, 154 anotaciones, exportación COCO,
  75 archivos Pascal VOC XML y 6 gráficas de análisis.
- data/dataset/data.yaml con ruta relativa `path: .`

## Verificaciones realizadas durante la preparación
- 75 imágenes JPEG legibles, todas de 640 × 640 px.
- Correspondencia de 75 imágenes con 75 etiquetas YOLO.
- 154 instancias en YOLO, 154 en COCO y 154 en Pascal VOC.
- Cuatro pruebas de humo aprobadas en ejecución local fuera de Docker.
- Evaluación local CPU: 15 imágenes, 35 señales, TP=35, FP=0, FN=0,
  P=1.000, R=1.000, F1=1.000, mAP@0.5=1.000, IoU medio=0.916.
  Esta prueba local no sustituye la evidencia de Docker Desktop.

## Pendiente antes de la entrega final
1. Ejecutar scripts/construir_y_ejecutar.ps1 en Windows con Docker Desktop
   (o scripts/construir_y_ejecutar.sh en Linux/macOS).
2. Comprobar las cuatro imágenes externas y revisar output/resumen_para_reporte.txt.
3. Incorporar las capturas reales de construcción y ejecución en la sección 6.8.
4. Publicar el repositorio en GitHub y sustituir los marcadores de URL en el reporte.

## No incluido deliberadamente
- output/: resultados y logs generados localmente; no se versiona.
- La imagen Docker: se construye y conserva en el equipo, no se adjunta.
- El notebook de Colab: no forma parte del entregable de la Actividad 4.3.
- El video demostrativo: el reporte técnico puede sustituirlo según lo aclarado
  por el docente.

No se inventaron capturas ni resultados de ejecución en Docker Desktop.
