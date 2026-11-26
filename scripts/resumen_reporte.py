#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Junta los resultados de output/ en un solo archivo con los datos que van en las
secciones 6.7 y 6.8 del reporte (tablas 10 y 11).

Lo llama construir_y_ejecutar.ps1 / .sh dentro del contenedor:
  python scripts/resumen_reporte.py --salida output
"""
import argparse
import json
import re
import sys
from pathlib import Path


def leer_json(ruta):
    try:
        return json.loads(Path(ruta).read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def leer_texto(ruta):
    p = Path(ruta)
    if not p.exists():
        return ""
    datos = p.read_bytes()
    for cod in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return datos.decode(cod)
        except UnicodeDecodeError:
            continue
    return ""


def top(det):
    if not det:
        return "sin detección", None
    d = max(det, key=lambda x: x["confianza"])
    return d["clase"], round(d["confianza"] * 100, 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="output")
    args = ap.parse_args(argv)
    out = Path(args.salida)
    L = []
    w = L.append

    info = leer_json(out / "logs" / "info_equipo.json") or {}
    ev = leer_json(out / "evaluacion" / "metricas.json") or {}
    pipe = leer_json(out / "resultados.json") or []
    video = leer_json(out / "video" / "estadisticas_video.json") or {}
    pruebas = leer_texto(out / "logs" / "02_pruebas.txt")
    m = re.search(r"(\d+) de (\d+) pruebas aprobadas", pruebas)

    w("=" * 72)
    w("  DATOS PARA EL REPORTE · Actividad 4.3")
    w("=" * 72)
    w("")
    w("TABLA 11 · Construcción y ejecución en el equipo del alumno (sección 6.8)")
    w("-" * 72)
    w(f"Sistema operativo        : {info.get('so', '?')}")
    w(f"Procesador y memoria     : {info.get('cpu', '?')} · {info.get('nucleos', '?')} núcleos · {info.get('ram_gb', '?')} GB de RAM")
    w(f"GPU                      : {info.get('gpu', '?')} (el contenedor usa CPU)")
    w(f"Docker Desktop           : {info.get('docker', '?')}")
    w("Comando de construcción  : docker build -t vision-senales-transito:1.0 .  (sin argumentos)")
    w(f"Tiempo de construcción   : {info.get('build_s', '?')} s")
    w(f"Tamaño de la imagen      : {info.get('tam_gb', '?')} GB")
    w(f"Pruebas de humo          : {m.group(1) + ' de ' + m.group(2) + ' aprobadas' if m else 'revisar output/logs/02_pruebas.txt'}")
    if ev:
        w(f"Evaluación               : P {ev['precision']:.3f} · R {ev['recall']:.3f} · F1 {ev['f1']:.3f} · "
          f"mAP@0.5 {ev['mAP50']:.3f} · IoU {ev['iou_medio']:.3f}  (TP {ev['TP']} / FP {ev['FP']} / FN {ev['FN']})")
    else:
        w("Evaluación               : no se encontró output/evaluacion/metricas.json")
    tiempos = [r["tiempos_ms"]["total"] for r in pipe if not r.get("error")]
    if tiempos:
        w(f"Tiempo por imagen en CPU : {sum(tiempos) / len(tiempos):.1f} ms de media "
          f"(entre {min(tiempos):.1f} y {max(tiempos):.1f} ms), {len(tiempos)} imágenes")
    if video:
        w(f"Video (extra)            : {video['frames']} cuadros · {video['fps_procesamiento']} FPS · "
          f"latencia media {video['latencia_media_ms']} ms · p95 {video['latencia_p95_ms']} ms · "
          f"tiempo real: {'sí' if video['tiempo_real'] else 'no'}")

    w("")
    w("TABLA 10 · Imágenes descargadas de internet (sección 6.7)")
    w("-" * 72)
    r06 = {Path(r["fuente"]).stem: r for r in (leer_json(out / "externas" / "resultados.json") or [])}
    r03 = {Path(r["fuente"]).stem: r for r in (leer_json(out / "externas_03" / "resultados.json") or [])}
    aciertos06 = aciertos03 = 0
    nombres = sorted(set(r06) | set(r03))
    w(f"{'Imagen':<18}{'Clase real':<15}{'Umbral 0.6':<26}{'Umbral 0.3':<26}¿Acierta?")
    for n in nombres:
        a = r06.get(n, {})
        b = r03.get(n, {})
        c6, p6 = top(a.get("detecciones")) if not a.get("error") else ("error", None)
        c3, p3 = top(b.get("detecciones")) if not b.get("error") else ("error", None)
        ok6, ok3 = c6 == n, c3 == n
        aciertos06 += ok6
        aciertos03 += ok3
        t6 = f"{c6} {p6} %" if p6 is not None else c6
        t3 = f"{c3} {p3} %" if p3 is not None else c3
        archivo = Path(a.get("fuente") or b.get("fuente") or n).name
        w(f"{archivo:<18}{n:<15}{t6:<26}{t3:<26}{'sí (0.6)' if ok6 else ('solo con 0.3' if ok3 else 'no')}")
    if nombres:
        w("")
        w(f"Aciertos: {aciertos06} de {len(nombres)} con umbral 0.6 y {aciertos03} de {len(nombres)} con 0.3.")

    w("")
    w("TEXTO SUGERIDO PARA LOS PÁRRAFOS CON HUECOS")
    w("-" * 72)
    if nombres:
        w(f"6.7: Con el umbral de operación el contenedor acertó {aciertos06} de {len(nombres)} imágenes "
          f"y con 0.3 acertó {aciertos03} de {len(nombres)}.")
    if info.get("tam_gb") and ev:
        w(f"6.8: La imagen ocupó {info['tam_gb']} GB, frente a los 9.84 GB de la imagen de prueba que instaló "
          f"PyTorch desde PyPI. Las métricas de evaluación "
          f"{'coincidieron' if (ev['f1'] == 1.0 and ev['mAP50'] == 1.0 and abs(ev['iou_medio'] - 0.916) < 0.0015) else 'no coincidieron exactamente'} "
          f"con las de la Tabla 9.")
    w("")
    w("Capturas: los textos de cada paso quedaron en output/logs/ y las imágenes anotadas en output/, "
      "output/externas y output/externas_03.")

    texto = "\n".join(L)
    (out / "resumen_para_reporte.txt").write_text(texto + "\n", encoding="utf-8")
    print(texto)
    return 0


if __name__ == "__main__":
    sys.exit(main())
