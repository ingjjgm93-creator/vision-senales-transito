#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================================
  MAESTRÍA EN INTELIGENCIA ARTIFICIAL - TecNM Virtual
  Visión Inteligente · Unidad IV. Despliegue de un Modelo de Visión Inteligente
  Actividad 4.1: Creación de un Pipeline Operativo

  pipeline_inferencia.py
  Pipeline de inferencia para detección de señales de tránsito con el modelo
  Faster R-CNN (MobileNetV3-Large FPN) entrenado con el dataset propio de la
  Actividad 1.4 y validado en la Actividad 3.4.

  Etapas:  entrada -> preprocesamiento -> inferencia -> postprocesamiento -> salida

  Uso desde la terminal:
    python pipeline_inferencia.py --modelo modelo_senales_frcnn.pth --entrada foto.jpg
    python pipeline_inferencia.py --modelo modelo_senales_frcnn.pth --entrada carpeta/ --lote 4
    python pipeline_inferencia.py --modelo modelo_senales_frcnn.pth --entrada video.mp4
    python pipeline_inferencia.py --modelo modelo_senales_frcnn.pth --entrada 0   (cámara web)
===========================================================================
"""
import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import batched_nms

__version__ = "1.1.0"

CLASES_DEFAULT = ["alto", "ceda_el_paso", "no_estacionar", "velocidad"]
# Colores BGR por clase (mismos de la Actividad 1.4: azul, cyan, rojo, verde)
COLORES = {"alto": (255, 80, 0), "ceda_el_paso": (255, 255, 0),
           "no_estacionar": (0, 0, 255), "velocidad": (0, 200, 0)}
EXT_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
EXT_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

log = logging.getLogger("pipeline")


def configurar_logs(archivo="pipeline.log", nivel=logging.INFO):
    """Registra los eventos del pipeline en consola y en un archivo de log."""
    log.setLevel(nivel)
    if log.handlers:
        return log
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    consola = logging.StreamHandler(sys.stdout)
    consola.setFormatter(fmt)
    log.addHandler(consola)
    if archivo:
        fh = logging.FileHandler(archivo, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log


# ---------------------------------------------------------------------------
# Estructuras de salida
# ---------------------------------------------------------------------------
@dataclass
class Deteccion:
    clase: str
    clase_id: int
    confianza: float
    caja: list  # [x1, y1, x2, y2] en píxeles de la imagen original


@dataclass
class Resultado:
    fuente: str
    ancho: int
    alto: int
    detecciones: list = field(default_factory=list)
    tiempos_ms: dict = field(default_factory=dict)
    error: str = None

    def a_dict(self):
        d = asdict(self)
        d["num_detecciones"] = len(self.detecciones)
        return d


# ---------------------------------------------------------------------------
# Construcción y carga del modelo
# ---------------------------------------------------------------------------
def construir_modelo(num_clases, pesos_coco=False):
    """Faster R-CNN MobileNetV3-Large-320 FPN con cabeza de num_clases + fondo.

    box_score_thresh=0.001 deja pasar detecciones de baja confianza; el filtrado
    real lo hace el postprocesamiento con el umbral configurado.
    """
    if pesos_coco:
        from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
        modelo = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
            box_score_thresh=0.001, box_detections_per_img=100)
    else:
        modelo = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None, weights_backbone=None,
            box_score_thresh=0.001, box_detections_per_img=100)
    in_features = modelo.roi_heads.box_predictor.cls_score.in_features
    modelo.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_clases + 1)
    return modelo


def guardar_modelo(modelo, ruta, clases, extra=None):
    """Guarda el checkpoint estandarizado (.pth) con pesos y metadatos."""
    ckpt = {
        "arquitectura": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "clases": list(clases),
        "state_dict": modelo.state_dict(),
        "version_torch": torch.__version__,
        "version_pipeline": __version__,
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, ruta)
    return ruta


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class PipelineSenales:
    """Pipeline operativo: carga del modelo, preprocesamiento, inferencia y postprocesamiento."""

    def __init__(self, ruta_modelo, dispositivo=None, umbral_conf=0.6, umbral_nms=0.5,
                 tam_max=640, usar_clahe=False, calentar=True):
        if dispositivo is None:
            dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dispositivo)
        self.umbral_conf = float(umbral_conf)
        self.umbral_nms = float(umbral_nms)
        self.tam_max = int(tam_max)
        self.usar_clahe = bool(usar_clahe)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.modelo, self.clases, self.meta = self.cargar_modelo(ruta_modelo)
        log.info("Pipeline listo | modelo=%s | clases=%s | dispositivo=%s | conf=%.2f | clahe=%s",
                 Path(ruta_modelo).name, self.clases, self.device, self.umbral_conf, self.usar_clahe)
        if calentar:
            self._calentar()

    # ---- 1. Carga del modelo ------------------------------------------------
    def cargar_modelo(self, ruta):
        ruta = Path(ruta)
        if not ruta.exists():
            raise FileNotFoundError(f"No existe el archivo del modelo: {ruta}")
        t0 = time.perf_counter()
        if ruta.suffix == ".pt":  # TorchScript
            extra = {"meta.json": ""}
            modelo = torch.jit.load(str(ruta), map_location=self.device, _extra_files=extra)
            meta = json.loads(extra["meta.json"] or "{}")
            clases = meta.get("clases", CLASES_DEFAULT)
        else:  # checkpoint .pth
            ckpt = torch.load(str(ruta), map_location=self.device, weights_only=False)
            if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
                raise ValueError("El archivo no es un checkpoint válido del pipeline (falta 'state_dict').")
            if ckpt.get("arquitectura") != "fasterrcnn_mobilenet_v3_large_320_fpn":
                raise ValueError(f"Arquitectura no soportada: {ckpt.get('arquitectura')}")
            clases = ckpt.get("clases", CLASES_DEFAULT)
            modelo = construir_modelo(len(clases))
            faltan, sobran = modelo.load_state_dict(ckpt["state_dict"], strict=False)
            # Al construir sin pesos COCO se usa BatchNorm2d en vez de FrozenBatchNorm2d;
            # la única diferencia es el contador num_batches_tracked (no afecta la inferencia).
            faltan = [k for k in faltan if not k.endswith("num_batches_tracked")]
            if faltan or sobran:
                raise RuntimeError(f"Pesos incompatibles. Faltan: {faltan[:5]} | Sobran: {sobran[:5]}")
            meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
        modelo.to(self.device).eval()
        log.info("Modelo cargado en %.0f ms (%s)", (time.perf_counter() - t0) * 1000, ruta.suffix)
        return modelo, list(clases), meta

    def _calentar(self):
        """Primera inferencia de prueba para que las mediciones de tiempo no incluyan la inicialización."""
        dummy = np.zeros((320, 320, 3), np.uint8)
        self.procesar_imagen(dummy, fuente="calentamiento")

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    # ---- 2. Entrada ---------------------------------------------------------
    @staticmethod
    def leer_imagen(ruta):
        ruta = Path(ruta)
        if not ruta.exists():
            raise FileNotFoundError(f"No existe la imagen: {ruta}")
        if ruta.suffix.lower() not in EXT_IMAGEN:
            raise ValueError(f"Formato de imagen no soportado: {ruta.suffix}")
        img = cv2.imdecode(np.fromfile(str(ruta), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"No se pudo decodificar la imagen (archivo dañado): {ruta}")
        return img

    # ---- 3. Preprocesamiento --------------------------------------------------
    def preprocesar(self, img):
        """Imagen BGR de OpenCV -> tensor RGB float [0,1] (C,H,W) + factor de escala.

        Pasos: validar tipo y canales, ecualización CLAHE opcional (canal L de Lab),
        redimensionar conservando proporción si el lado mayor supera tam_max,
        BGR->RGB y normalización a [0,1]. La normalización por media/desviación y el
        redimensionamiento a 320 px los aplica internamente el GeneralizedRCNNTransform.
        """
        if not isinstance(img, np.ndarray):
            raise TypeError("La entrada debe ser un arreglo de NumPy (imagen de OpenCV).")
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        h, w = img.shape[:2]
        if h < 16 or w < 16:
            raise ValueError(f"Imagen demasiado pequeña: {w}x{h}")
        if self.usar_clahe:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        escala = 1.0
        if max(h, w) > self.tam_max:
            escala = self.tam_max / max(h, w)
            img = cv2.resize(img, (round(w * escala), round(h * escala)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
        return tensor, escala

    # ---- 4. Inferencia --------------------------------------------------------
    @torch.inference_mode()
    def inferir(self, tensores):
        """Recibe una lista de tensores (un lote) y devuelve las salidas crudas del modelo en CPU."""
        entrada = [t.to(self.device, non_blocking=True) for t in tensores]
        salida = self.modelo(entrada)
        if isinstance(salida, tuple):  # TorchScript devuelve (pérdidas, detecciones)
            salida = salida[1]
        return [{k: v.detach().cpu() for k, v in s.items()} for s in salida]

    # ---- 5. Postprocesamiento -------------------------------------------------
    def postprocesar(self, salida, escala, ancho, alto):
        """Filtra por confianza, aplica NMS por clase, regresa las cajas a la escala original."""
        boxes, scores, labels = salida["boxes"], salida["scores"], salida["labels"]
        keep = scores >= self.umbral_conf
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        if len(boxes):
            keep = batched_nms(boxes, scores, labels, self.umbral_nms)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        boxes = boxes / escala
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, ancho)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, alto)
        dets = []
        for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
            if 1 <= l <= len(self.clases):
                dets.append(Deteccion(self.clases[l - 1], int(l), round(float(s), 4),
                                      [round(v, 1) for v in b]))
        dets.sort(key=lambda d: -d.confianza)
        return dets

    # ---- Flujos completos -----------------------------------------------------
    def procesar_imagen(self, imagen, fuente="memoria"):
        """Una imagen (ruta o arreglo BGR) -> Resultado. Los errores se registran, no detienen el flujo."""
        try:
            t0 = time.perf_counter()
            if isinstance(imagen, (str, Path)):
                fuente = str(imagen)
                imagen = self.leer_imagen(imagen)
            alto, ancho = imagen.shape[:2]
            tensor, escala = self.preprocesar(imagen)
            t1 = time.perf_counter()
            salida = self.inferir([tensor])[0]
            self._sync()
            t2 = time.perf_counter()
            dets = self.postprocesar(salida, escala, ancho, alto)
            t3 = time.perf_counter()
            tiempos = {"preprocesamiento": (t1 - t0) * 1000, "inferencia": (t2 - t1) * 1000,
                       "postprocesamiento": (t3 - t2) * 1000, "total": (t3 - t0) * 1000}
            return Resultado(fuente, ancho, alto, dets, {k: round(v, 2) for k, v in tiempos.items()})
        except Exception as e:  # gestión de errores
            log.error("Error procesando %s: %s", fuente, e)
            return Resultado(fuente, 0, 0, [], {}, error=f"{type(e).__name__}: {e}")

    def procesar_lote(self, imagenes, tam_lote=4):
        """Inferencia por lotes: agrupa tam_lote imágenes en una sola pasada del modelo."""
        resultados = []
        for i in range(0, len(imagenes), tam_lote):
            grupo = imagenes[i:i + tam_lote]
            validos, fallidos = [], []
            t0 = time.perf_counter()
            for item in grupo:
                fuente = str(item) if isinstance(item, (str, Path)) else f"lote_{i}"
                try:
                    img = self.leer_imagen(item) if isinstance(item, (str, Path)) else item
                    alto, ancho = img.shape[:2]
                    tensor, escala = self.preprocesar(img)
                    validos.append((fuente, tensor, escala, ancho, alto))
                except Exception as e:
                    log.error("Error en %s: %s", fuente, e)
                    fallidos.append(Resultado(fuente, 0, 0, [], {}, error=f"{type(e).__name__}: {e}"))
            t1 = time.perf_counter()
            salidas = self.inferir([v[1] for v in validos]) if validos else []
            self._sync()
            t2 = time.perf_counter()
            parciales = [(v, self.postprocesar(s, v[2], v[3], v[4])) for v, s in zip(validos, salidas)]
            t3 = time.perf_counter()
            n = max(len(validos), 1)
            tiempos = {"preprocesamiento": (t1 - t0) * 1000 / n, "inferencia": (t2 - t1) * 1000 / n,
                       "postprocesamiento": (t3 - t2) * 1000 / n, "total": (t3 - t0) * 1000 / n,
                       "lote_total": (t3 - t0) * 1000, "tam_lote": len(validos)}
            for v, dets in parciales:
                resultados.append(Resultado(v[0], v[3], v[4], dets, {k: round(x, 2) for k, x in tiempos.items()}))
            resultados.extend(fallidos)
            log.info("Lote %d: %d imágenes en %.1f ms (%.1f ms/imagen)", i // tam_lote + 1,
                     len(validos), tiempos["lote_total"], tiempos["total"])
        return resultados

    def procesar_video(self, fuente, salida_video=None, max_frames=None, cada_n=1, callback=None):
        """Inferencia en tiempo real cuadro por cuadro desde un video, una cámara (índice) o una URL de stream.

        Devuelve estadísticas de latencia y FPS. Si salida_video se indica, guarda el video anotado.
        """
        src = int(fuente) if str(fuente).isdigit() else str(fuente)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise IOError(f"No se pudo abrir la fuente de video: {fuente}")
        fps_fuente = cap.get(cv2.CAP_PROP_FPS) or 15.0
        escritor, latencias, conteo, n, ultimo = None, [], {}, 0, None
        t_inicio = time.perf_counter()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                n += 1
                if n % cada_n == 0 or ultimo is None:
                    ultimo = self.procesar_imagen(frame, fuente=f"{fuente}#frame{n}")
                    if ultimo.error is None:
                        latencias.append(ultimo.tiempos_ms["total"])
                        for d in ultimo.detecciones:
                            conteo[d.clase] = conteo.get(d.clase, 0) + 1
                anotado = dibujar(frame, ultimo, mostrar_fps=1000 / max(ultimo.tiempos_ms.get("total", 1), 1e-3))
                if salida_video:
                    if escritor is None:
                        h, w = anotado.shape[:2]
                        escritor = cv2.VideoWriter(str(salida_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                                   fps_fuente, (w, h))
                    escritor.write(anotado)
                if callback:
                    callback(n, anotado, ultimo)
                if max_frames and n >= max_frames:
                    break
        finally:
            cap.release()
            if escritor is not None:
                escritor.release()
        t_total = time.perf_counter() - t_inicio
        lat = np.array(latencias) if latencias else np.array([0.0])
        stats = {"fuente": str(fuente), "frames": n, "frames_inferidos": len(latencias),
                 "fps_fuente": round(fps_fuente, 2), "fps_procesamiento": round(n / t_total, 2) if t_total else 0,
                 "latencia_media_ms": round(float(lat.mean()), 2),
                 "latencia_p95_ms": round(float(np.percentile(lat, 95)), 2),
                 "latencia_max_ms": round(float(lat.max()), 2),
                 "tiempo_real": bool(n / t_total >= fps_fuente) if t_total else False,
                 "detecciones_por_clase": conteo}
        log.info("Video %s: %d frames | %.1f FPS de procesamiento | latencia media %.1f ms",
                 fuente, n, stats["fps_procesamiento"], stats["latencia_media_ms"])
        return stats


# ---------------------------------------------------------------------------
# Salidas
# ---------------------------------------------------------------------------
def dibujar(imagen, resultado, mostrar_fps=None):
    """Dibuja cajas, clase y confianza sobre una copia de la imagen BGR."""
    out = imagen.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    if resultado is None:
        return out
    grosor = max(2, round(max(out.shape[:2]) / 320))
    for d in resultado.detecciones:
        x1, y1, x2, y2 = map(int, d.caja)
        color = COLORES.get(d.clase, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, grosor)
        texto = f"{d.clase} {d.confianza:.2f}"
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_txt = max(y1, th + 6)
        cv2.rectangle(out, (x1, y_txt - th - 6), (x1 + tw + 4, y_txt), color, -1)
        cv2.putText(out, texto, (x1 + 2, y_txt - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    if mostrar_fps is not None:
        cv2.putText(out, f"{mostrar_fps:.1f} FPS | {len(resultado.detecciones)} senales", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, f"{mostrar_fps:.1f} FPS | {len(resultado.detecciones)} senales", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def guardar_resultados(resultados, carpeta, guardar_imagenes=True):
    """Guarda resultados.json y, opcionalmente, cada imagen anotada."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    for r in resultados:
        if guardar_imagenes and r.error is None and Path(r.fuente).exists():
            img = PipelineSenales.leer_imagen(r.fuente)
            cv2.imwrite(str(carpeta / f"{Path(r.fuente).stem}_det.jpg"), dibujar(img, r))
    with open(carpeta / "resultados.json", "w", encoding="utf-8") as f:
        json.dump([r.a_dict() for r in resultados], f, indent=2, ensure_ascii=False)
    return carpeta / "resultados.json"


# ---------------------------------------------------------------------------
# Interfaz de línea de comandos
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Pipeline de inferencia - Detección de señales de tránsito (Faster R-CNN)")
    ap.add_argument("--modelo", required=True, help="Ruta del modelo (.pth checkpoint o .pt TorchScript)")
    ap.add_argument("--entrada", required=True, help="Imagen, carpeta de imágenes, video o índice de cámara (0)")
    ap.add_argument("--salida", default="salidas", help="Carpeta de resultados")
    ap.add_argument("--conf", type=float, default=0.6, help="Umbral de confianza (0.6 = mejor F1 en la Act. 3.4)")
    ap.add_argument("--nms", type=float, default=0.5, help="Umbral IoU para NMS")
    ap.add_argument("--lote", type=int, default=4, help="Tamaño de lote para carpetas")
    ap.add_argument("--clahe", action="store_true", help="Aplicar ecualización CLAHE en el preprocesamiento")
    ap.add_argument("--dispositivo", default=None, help="cpu o cuda (por defecto se detecta)")
    ap.add_argument("--max-frames", type=int, default=None, help="Límite de cuadros para video/cámara")
    args = ap.parse_args(argv)

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    configurar_logs(salida / "pipeline.log")
    try:
        pipe = PipelineSenales(args.modelo, args.dispositivo, args.conf, args.nms, usar_clahe=args.clahe)
    except Exception as e:
        log.critical("No se pudo iniciar el pipeline: %s", e)
        return 1

    entrada = Path(args.entrada)
    if args.entrada.isdigit() or entrada.suffix.lower() in EXT_VIDEO or args.entrada.startswith(("rtsp://", "http")):
        try:
            stats = pipe.procesar_video(args.entrada, salida / "video_anotado.mp4", args.max_frames)
        except Exception as e:
            log.error("%s", e)
            return 1
        with open(salida / "estadisticas_video.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return 0

    if entrada.is_dir():
        rutas = sorted(p for p in entrada.iterdir() if p.suffix.lower() in EXT_IMAGEN)
        if not rutas:
            log.error("La carpeta no contiene imágenes: %s", entrada)
            return 1
        resultados = pipe.procesar_lote(rutas, args.lote)
    else:
        resultados = [pipe.procesar_imagen(entrada)]

    ruta_json = guardar_resultados(resultados, salida)
    for r in resultados:
        if r.error:
            print(f"✗ {Path(r.fuente).name}: {r.error}")
        else:
            dets = ", ".join(f"{d.clase} ({d.confianza:.2f})" for d in r.detecciones) or "sin detecciones"
            print(f"✓ {Path(r.fuente).name}: {dets} | {r.tiempos_ms['total']:.1f} ms")
    print(f"Resultados guardados en: {ruta_json}")
    return 0 if all(r.error is None for r in resultados) else 2


if __name__ == "__main__":
    sys.exit(main())