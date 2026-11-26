#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visión Inteligente · Unidad IV · Actividad 4.3
crear_video_demo.py

Genera un video de cámara simulada con las imágenes de validación del dataset
para probar el modo de tiempo real sin necesidad de una cámara:
15 imágenes x 4 cuadros = 60 cuadros a 15 FPS, 640x640 px.

  python src/crear_video_demo.py
  python src/crear_video_demo.py --cuadros 4 --fps 15 --salida data/video/camara_simulada.mp4
"""
import argparse
import sys
from pathlib import Path

import cv2

RAIZ = Path(__file__).resolve().parent.parent
EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Crea el video de cámara simulada")
    ap.add_argument("--imagenes", default=str(RAIZ / "data" / "dataset" / "images" / "val"))
    ap.add_argument("--salida", default=str(RAIZ / "data" / "video" / "camara_simulada.mp4"))
    ap.add_argument("--cuadros", type=int, default=4, help="Cuadros por imagen")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--tam", type=int, default=640)
    args = ap.parse_args(argv)

    rutas = sorted(p for p in Path(args.imagenes).iterdir() if p.suffix.lower() in EXT)
    if not rutas:
        print(f"No hay imágenes en {args.imagenes}")
        return 1
    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(salida), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.tam, args.tam))
    n = 0
    for ruta in rutas:
        img = cv2.resize(cv2.imread(str(ruta)), (args.tam, args.tam))
        for _ in range(args.cuadros):
            vw.write(img)
            n += 1
    vw.release()
    print(f"Video creado: {salida} | {n} cuadros a {args.fps} FPS ({n / args.fps:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
