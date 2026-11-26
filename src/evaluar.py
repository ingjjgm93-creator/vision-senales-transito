#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================================
  Visión Inteligente · Unidad IV · Actividad 4.3
  evaluar.py

  Compara las detecciones del pipeline con las etiquetas YOLO del dataset
  propio (Unidad 1) y calcula, con IoU 0.5:
    - TP, FP, FN, precisión, recall y F1 en el umbral de operación (0.6)
    - AP@0.5 por clase y mAP@0.5 (con todas las detecciones de confianza >= 0.05)
    - IoU medio de los verdaderos positivos

  Uso:
    python src/evaluar.py
    python src/evaluar.py --split val --salida output/evaluacion
===========================================================================
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_inferencia import EXT_IMAGEN, PipelineSenales, configurar_logs  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent


def leer_etiquetas(dataset, split):
    """Imágenes y cajas reales del split. Las etiquetas YOLO se pasan a píxeles [x1, y1, x2, y2]."""
    import cv2
    img_dir = Path(dataset) / "images" / split
    lbl_dir = Path(dataset) / "labels" / split
    if not img_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta del dataset: {img_dir}")
    items = []
    for f in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in EXT_IMAGEN):
        h, w = cv2.imread(str(f)).shape[:2]
        cajas, clases = [], []
        lbl = lbl_dir / (f.stem + ".txt")
        if lbl.exists():
            for linea in lbl.read_text(encoding="utf-8").splitlines():
                p = linea.split()
                if len(p) < 5:
                    continue
                c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
                cajas.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
                clases.append(c + 1)  # el modelo usa 0 para el fondo
        items.append({"ruta": f, "cajas": np.array(cajas, np.float64).reshape(-1, 4), "clases": np.array(clases, int)})
    return items


def iou(caja, cajas):
    if len(cajas) == 0:
        return np.array([])
    x1 = np.maximum(caja[0], cajas[:, 0]); y1 = np.maximum(caja[1], cajas[:, 1])
    x2 = np.minimum(caja[2], cajas[:, 2]); y2 = np.minimum(caja[3], cajas[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (caja[2] - caja[0]) * (caja[3] - caja[1])
    b = (cajas[:, 2] - cajas[:, 0]) * (cajas[:, 3] - cajas[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)


def emparejar(resultados, items, conf_min, iou_thr):
    """Empareja detecciones con cajas reales de la misma clase (mayor confianza primero)."""
    registros, ious, npos = [], [], {}
    for r, m in zip(resultados, items):
        for c in m["clases"]:
            npos[int(c)] = npos.get(int(c), 0) + 1
        usado = np.zeros(len(m["clases"]), bool)
        for d in sorted(r.detecciones, key=lambda d: -d.confianza):
            if d.confianza < conf_min:
                continue
            v = iou(np.array(d.caja), m["cajas"])
            if len(v):
                v[(m["clases"] != d.clase_id) | usado] = 0
            j = int(np.argmax(v)) if len(v) else -1
            ok = j >= 0 and v[j] >= iou_thr
            if ok:
                usado[j] = True
                ious.append(float(v[j]))
            registros.append((d.clase_id, d.confianza, ok))
    return registros, ious, npos


def average_precision(registros, npos, clase):
    regs = sorted([r for r in registros if r[0] == clase], key=lambda r: -r[1])
    n = npos.get(clase, 0)
    if n == 0:
        return float("nan")
    tp = np.cumsum([1 if r[2] else 0 for r in regs]) if regs else np.array([0])
    fp = np.cumsum([0 if r[2] else 1 for r in regs]) if regs else np.array([0])
    rec = tp / n
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluación del pipeline contra las etiquetas YOLO del dataset propio")
    ap.add_argument("--modelo", default=str(RAIZ / "models" / "modelo_senales_frcnn.pth"))
    ap.add_argument("--dataset", default=str(RAIZ / "data" / "dataset"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.6, help="Umbral de operación para P, R y F1")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--salida", default="output/evaluacion")
    ap.add_argument("--dispositivo", default=None)
    args = ap.parse_args(argv)

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    configurar_logs(salida / "evaluacion.log")

    items = leer_etiquetas(args.dataset, args.split)
    # Umbral bajo para que el AP vea toda la curva; P, R y F1 se calculan después con args.conf
    pipe = PipelineSenales(args.modelo, args.dispositivo, umbral_conf=0.05)
    resultados = pipe.procesar_lote([m["ruta"] for m in items], tam_lote=4)
    errores = [r for r in resultados if r.error]
    if errores:
        print(f"Atención: {len(errores)} imágenes con error de lectura")

    regs_op, ious, npos = emparejar(resultados, items, args.conf, args.iou)
    tp = sum(1 for r in regs_op if r[2])
    fp = len(regs_op) - tp
    total_real = sum(npos.values())
    fn = total_real - tp
    p = tp / max(tp + fp, 1)
    r = tp / max(total_real, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)

    regs_ap, _, _ = emparejar(resultados, items, 0.05, args.iou)
    ap_clase = {c: average_precision(regs_ap, npos, i + 1) for i, c in enumerate(pipe.clases)}
    validos = [v for v in ap_clase.values() if not np.isnan(v)]
    tiempos = [x.tiempos_ms["total"] for x in resultados if x.error is None]

    metricas = {
        "split": args.split, "imagenes": len(items), "senales_reales": total_real,
        "umbral_conf": args.conf, "umbral_iou": args.iou,
        "TP": tp, "FP": fp, "FN": fn,
        "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
        "AP50_por_clase": {c: round(v, 3) for c, v in ap_clase.items()},
        "mAP50": round(float(np.mean(validos)), 3) if validos else 0.0,
        "iou_medio": round(float(np.mean(ious)), 3) if ious else 0.0,
        "ms_por_imagen": round(float(np.mean(tiempos)), 1) if tiempos else 0.0,
        "dispositivo": str(pipe.device),
    }
    (salida / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 64)
    print(f"  Evaluación · split {args.split} · {len(items)} imágenes · {total_real} señales reales")
    print("=" * 64)
    print(f"  Umbral de confianza {args.conf} · IoU {args.iou}")
    print(f"  TP {tp} | FP {fp} | FN {fn}")
    print(f"  Precisión {p:.3f} | Recall {r:.3f} | F1 {f1:.3f}")
    for c, v in ap_clase.items():
        print(f"  AP@0.5 {c:<15} {v:.3f}")
    print(f"  mAP@0.5 {metricas['mAP50']:.3f} | IoU medio {metricas['iou_medio']:.3f}")
    print(f"  Tiempo medio por imagen: {metricas['ms_por_imagen']} ms ({pipe.device})")
    print(f"  Resultados en: {salida / 'metricas.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
