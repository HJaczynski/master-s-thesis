param(
    [switch]$WithTensorRT
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Python = ".\venv\Scripts\python.exe"
if (!(Test-Path $Python)) {
    throw "Virtual environment not found at .\venv. Create it before installing experiment dependencies."
}

& $Python -m pip install -U pip
& $Python -m pip install -U ultralytics onnx onnxruntime-gpu nvidia-ml-py

if ($WithTensorRT) {
    & $Python -m pip install -U --extra-index-url https://pypi.nvidia.com tensorrt
}

& $Python -m pip install -e . --no-deps --no-build-isolation
& .\venv\Scripts\cv-benchmark.exe doctor --profile smoke

Write-Host ""
Write-Host "Base RF-DETR/YOLO stack checked. For Swin Mask R-CNN, run:"
Write-Host "  .\scripts\install_swin_mmdetection.ps1"
