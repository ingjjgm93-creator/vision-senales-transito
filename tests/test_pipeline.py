#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visión Inteligente · Unidad IV · Actividad 4.3
test_pipeline.py — pruebas de humo del contenedor

  python tests/test_pipeline.py

1. El checkpoint carga con su arquitectura y sus 4 clases.
2. El detector encuentra señales en una imagen de validación con confianza >= 0.6.
3. Una imagen gris sin señales produce cero detecciones.
4. Tres entradas inválidas (inexistente, dañada y de formato no soportado) se
   registran como error sin detener el lote.

Devuelve código 0 si todo pasa y 1 si algo falla.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
from pipeline_inferencia import EXT_IMAGEN, PipelineSenales, configurar_logs  # noqa: E402

MODELO = RAIZ / "models" / "modelo_senales_frcnn.pth"
VAL = RAIZ / "data" / "dataset" / "images" / "val"
resultados = []


def prueba(nombre, ok, detalle=""):
    resultados.append(ok)
    print(f"[{'OK   ' if ok else 'FALLA'}] {nombre}" + (f" -> {detalle}" if detalle else ""))


def main():
    configurar_logs(None)
    try:
        pipe = PipelineSenales(MODELO, umbral_conf=0.6)
        ok = pipe.meta.get("arquitectura") == "fasterrcnn_mobilenet_v3_large_320_fpn" and len(pipe.clases) == 4
        prueba("1. Carga del checkpoint", ok, f"{pipe.meta.get('arquitectura')} | clases {pipe.clases}")
    except Exception as e:
        prueba("1. Carga del checkpoint", False, str(e))
        return 1

    imagenes = sorted(p for p in VAL.iterdir() if p.suffix.lower() in EXT_IMAGEN) if VAL.exists() else []
    if imagenes:
        r = pipe.procesar_imagen(imagenes[0])
        ok = r.error is None and len(r.detecciones) > 0 and all(d.confianza >= 0.6 for d in r.detecciones)
        dets = ", ".join(f"{d.clase} {d.confianza:.2f}" for d in r.detecciones) or "sin detecciones"
        prueba("2. Detección en imagen de validación", ok, f"{imagenes[0].name}: {dets}")
    else:
        prueba("2. Detección en imagen de validación", False, f"no hay imágenes en {VAL}")

    gris = np.full((480, 640, 3), 128, np.uint8)
    r = pipe.procesar_imagen(gris, fuente="gris")
    prueba("3. Imagen gris sin señales", r.error is None and len(r.detecciones) == 0,
           f"{len(r.detecciones)} detecciones")

    with tempfile.TemporaryDirectory() as tmp:
        danada = Path(tmp) / "danada.jpg"
        danada.write_bytes(b"esto no es una imagen")
        no_soportado = Path(tmp) / "archivo.gif"
        no_soportado.write_bytes(b"GIF89a")
        entradas = [Path(tmp) / "no_existe.jpg", danada, no_soportado]
        if imagenes:
            entradas.append(imagenes[0])
        rs = pipe.procesar_lote(entradas, tam_lote=4)
        con_error = sum(1 for x in rs if x.error)
        sin_error = sum(1 for x in rs if x.error is None)
        ok = con_error == 3 and len(rs) == len(entradas)
        prueba("4. Entradas inválidas sin detener el lote", ok,
               f"{con_error} con error, {sin_error} procesadas de {len(entradas)}")

    aprobadas = sum(resultados)
    print(f"\n{aprobadas} de {len(resultados)} pruebas aprobadas")
    return 0 if all(resultados) else 1


if __name__ == "__main__":
    sys.exit(main())
