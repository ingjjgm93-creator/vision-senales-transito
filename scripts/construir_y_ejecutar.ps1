# =============================================================================
#  Actividad 4.3 · Visión Inteligente · construir_y_ejecutar.ps1 (Windows + Docker Desktop)
#
#  Construye la imagen, corre las pruebas y todos los modos, y deja en
#  output\resumen_para_reporte.txt los datos de las tablas 10 y 11 del reporte.
#
#  Uso (PowerShell, en la carpeta del repositorio):
#     powershell -ExecutionPolicy Bypass -File scripts\construir_y_ejecutar.ps1
# =============================================================================
$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Raiz = Split-Path -Parent $PSScriptRoot
Set-Location $Raiz
$Imagen = "vision-senales-transito:1.0"
$Logs = Join-Path $Raiz "output\logs"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

function Titulo($texto) {
    Write-Host ""
    Write-Host ("=" * 76) -ForegroundColor Cyan
    Write-Host "  $texto" -ForegroundColor Cyan
    Write-Host ("=" * 76) -ForegroundColor Cyan
}

# Ejecuta un comando de Docker, lo muestra en pantalla y guarda el texto en output\logs
function Ejecutar($archivo, $comando) {
    Write-Host "PS> $comando" -ForegroundColor Yellow
    $global:LASTEXITCODE = 0
    Invoke-Expression "$comando 2>&1" | ForEach-Object { $l = "$_"; Write-Host $l; $l } |
        Out-File -Encoding utf8 (Join-Path $Logs $archivo)
    return $LASTEXITCODE
}

function Detener($mensaje) {
    Write-Host ""
    Write-Host "ERROR: $mensaje" -ForegroundColor Red
    exit 1
}

# ----------------------------------------------------------------------------- 0. Requisitos
Titulo "0. Revisión de requisitos"
docker info *> $null
if ($LASTEXITCODE -ne 0) { Detener "Docker no responde. Abre Docker Desktop y espera a que diga 'Engine running'." }
Write-Host "Docker Desktop está corriendo."

if (-not (Test-Path "models\modelo_senales_frcnn.pth")) {
    Detener "Falta models\modelo_senales_frcnn.pth (el modelo de la Actividad 4.1). Cópialo a la carpeta models\."
}
Write-Host ("Modelo: models\modelo_senales_frcnn.pth ({0:N1} MB)" -f ((Get-Item "models\modelo_senales_frcnn.pth").Length / 1MB))

# Dataset: si no está en data\dataset, se busca dataset.zip en la carpeta del repositorio
if (-not (Test-Path "data\dataset\images\val")) {
    $zip = Get-ChildItem -Path $Raiz -Filter "dataset*.zip" -File | Select-Object -First 1
    if (-not $zip) { Detener "Falta el dataset. Copia images\, labels\ y data.yaml a data\dataset\ o deja dataset.zip en la carpeta del repositorio." }
    Write-Host "Extrayendo $($zip.Name) en data\dataset ..."
    $tmp = Join-Path $env:TEMP "dataset_act43"
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    Expand-Archive -Path $zip.FullName -DestinationPath $tmp -Force
    $base = Get-ChildItem -Path $tmp -Recurse -Directory -Filter "images" |
        Where-Object { Test-Path (Join-Path $_.FullName "val") } | Select-Object -First 1
    if (-not $base) { Detener "El zip no tiene la carpeta images\val." }
    $origen = $base.Parent.FullName
    New-Item -ItemType Directory -Force -Path "data\dataset" | Out-Null
    Copy-Item -Recurse -Force (Join-Path $origen "images") "data\dataset\"
    Copy-Item -Recurse -Force (Join-Path $origen "labels") "data\dataset\"
    $yaml = "path: .`ntrain: images/train`nval: images/val`nnc: 4`nnames:`n- alto`n- ceda_el_paso`n- no_estacionar`n- velocidad`n"
    [System.IO.File]::WriteAllText((Join-Path $Raiz "data\dataset\data.yaml"), $yaml)
}
$nVal = (Get-ChildItem "data\dataset\images\val" -File).Count
Write-Host "Dataset: data\dataset ($nVal imágenes de validación)"

# Imágenes de internet para la sección 6.7
New-Item -ItemType Directory -Force -Path "ejemplos" | Out-Null
$w = "https://upload.wikimedia.org/wikipedia/commons/thumb"
$externas = [ordered]@{
    "alto.jpg"          = "$w/7/71/ALTO_sign.jpg/500px-ALTO_sign.jpg"
    "ceda_el_paso.png"  = "$w/b/b9/Mexico_traffic_signal_sr7.svg/500px-Mexico_traffic_signal_sr7.svg.png"
    "no_estacionar.jpg" = "$w/1/1f/Se%C3%B1al_de_no_estacionarse.jpg/500px-Se%C3%B1al_de_no_estacionarse.jpg"
    "velocidad.png"     = "$w/2/2d/Mexico_road_sign_SR-09.svg/500px-Mexico_road_sign_SR-09.svg.png"
}
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
foreach ($nombre in $externas.Keys) {
    $destino = Join-Path "ejemplos" $nombre
    if (-not (Test-Path $destino)) {
        try {
            Invoke-WebRequest -Uri $externas[$nombre] -OutFile $destino -UserAgent "VisionInteligenteTecNM/1.0 (actividad academica)" -UseBasicParsing
            Write-Host "Descargada: ejemplos\$nombre"
            Start-Sleep -Seconds 1
        } catch { Write-Host "No se pudo descargar $nombre ($($_.Exception.Message)). Puedes copiar otra foto a ejemplos\." -ForegroundColor DarkYellow }
    }
}

# ----------------------------------------------------------------------------- 1. Datos del equipo
Titulo "1. Datos del equipo (Tabla 11)"
$so = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$gpu = (Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join ", "
$dockerVer = (docker version --format "{{.Server.Platform.Name}} (Engine {{.Server.Version}})") 2>$null
$info = [ordered]@{
    so      = "$($so.Caption) (compilación $($so.BuildNumber))"
    cpu     = $cpu.Name.Trim()
    nucleos = "$($cpu.NumberOfCores) núcleos / $($cpu.NumberOfLogicalProcessors) hilos"
    ram_gb  = [math]::Round($so.TotalVisibleMemorySize / 1MB, 1)
    gpu     = $gpu
    docker  = "$dockerVer"
}
$info.GetEnumerator() | ForEach-Object { Write-Host ("{0,-8}: {1}" -f $_.Key, $_.Value) }
docker version 2>&1 | Out-File -Encoding utf8 (Join-Path $Logs "00_docker_version.txt")

# ----------------------------------------------------------------------------- 2. Construcción
Titulo "2. Construcción de la imagen (Figura 12)  ·  tarda varios minutos la primera vez"
$inicio = Get-Date
$codigo = Ejecutar "01_build.txt" "docker build -t $Imagen ."
$info.build_s = [math]::Round(((Get-Date) - $inicio).TotalSeconds, 0)
if ($codigo -ne 0) { Detener "Falló docker build. Revisa output\logs\01_build.txt." }
Write-Host "Construcción terminada en $($info.build_s) s" -ForegroundColor Green

# Video de cámara simulada (se genera con el propio contenedor y se reconstruye en segundos)
if (-not (Test-Path "data\video\camara_simulada.mp4")) {
    New-Item -ItemType Directory -Force -Path "data\video" | Out-Null
    Ejecutar "01b_video_demo.txt" "docker run --rm --user root --entrypoint python -v `"${Raiz}\data:/app/data`" $Imagen src/crear_video_demo.py" | Out-Null
    Ejecutar "01c_rebuild.txt" "docker build -t $Imagen ." | Out-Null
}

Titulo "3. Imagen generada (Figura 13)"
Ejecutar "03_images.txt" "docker images vision-senales-transito" | Out-Null
$bytes = [double](docker image inspect $Imagen --format "{{.Size}}")
$info.tam_gb = [math]::Round($bytes / 1e9, 2)
Write-Host "Tamaño de la imagen: $($info.tam_gb) GB" -ForegroundColor Green
$info | ConvertTo-Json | Out-File -Encoding utf8 (Join-Path $Logs "info_equipo.json")

# ----------------------------------------------------------------------------- 4. Pruebas y modos
$vol = "-v `"${Raiz}\output:/app/output`""
Titulo "4. Pruebas de humo"
Ejecutar "02_pruebas.txt" "docker run --rm --entrypoint python $Imagen tests/test_pipeline.py" | Out-Null

Titulo "5. Modo por defecto: imágenes de validación"
Ejecutar "04_pipeline.txt" "docker run --rm $vol $Imagen" | Out-Null

Titulo "6. Evaluación contra las etiquetas YOLO"
Ejecutar "05_evaluacion.txt" "docker run --rm $vol --entrypoint python $Imagen src/evaluar.py --salida output/evaluacion" | Out-Null

Titulo "7. Video de cámara simulada"
Ejecutar "06_video.txt" "docker run --rm $vol $Imagen --entrada data/video/camara_simulada.mp4 --salida output/video" | Out-Null

$ej = "-v `"${Raiz}\ejemplos:/app/ejemplos:ro`""
Titulo "8. Imágenes de internet, umbral 0.6 (Figura 11 y Tabla 10)"
Ejecutar "07_externas_06.txt" "docker run --rm $ej $vol $Imagen --entrada ejemplos --salida output/externas" | Out-Null
Titulo "9. Imágenes de internet, umbral 0.3"
Ejecutar "08_externas_03.txt" "docker run --rm $ej $vol $Imagen --entrada ejemplos --conf 0.3 --salida output/externas_03" | Out-Null

# ----------------------------------------------------------------------------- 10. Resumen
Titulo "10. Datos para el reporte"
docker run --rm -v "${Raiz}\output:/app/output" -v "${Raiz}\scripts:/app/scripts:ro" --entrypoint python $Imagen scripts/resumen_reporte.py --salida output
Write-Host ""
Write-Host "Listo. Abre output\resumen_para_reporte.txt y copia los valores al reporte." -ForegroundColor Green
Write-Host "Imágenes anotadas: output\, output\externas\ y output\externas_03\  ·  textos de cada paso: output\logs\" -ForegroundColor Green
