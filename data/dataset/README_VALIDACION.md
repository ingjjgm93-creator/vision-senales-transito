# Dataset de señales de tránsito — Actividad 1.4

Dataset integrado desde el ZIP entregado para la Unidad I.

- 75 imágenes JPEG RGB de 640 × 640 px.
- 60 imágenes de entrenamiento y 15 de validación.
- 154 anotaciones: 119 en train y 35 en val.
- Clases: alto (39), ceda_el_paso (44), no_estacionar (39), velocidad (32).
- 75 archivos de etiquetas YOLO; 1 exportación COCO con 154 anotaciones.
- 75 archivos Pascal VOC XML; 6 gráficas de análisis.
- data.yaml utiliza `path: .` y rutas relativas para train y val.

En la inspección del ZIP no se detectaron imágenes dañadas, archivos de
etiquetas faltantes, líneas YOLO con formato inválido, referencias COCO
inexistentes ni cajas Pascal VOC inválidas.

La evaluación del modelo es una operación distinta de esta verificación
estructural y se documenta en el reporte de la Actividad 4.3.
