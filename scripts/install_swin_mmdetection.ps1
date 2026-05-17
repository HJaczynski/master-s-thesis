param(
    [switch]$SkipDoctor
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Python = ".\venv\Scripts\python.exe"
if (!(Test-Path $Python)) {
    throw "Virtual environment not found at .\venv. Create it before installing MMDetection."
}

& $Python -m pip install -U openmim
& .\venv\Scripts\mim.exe install "mmengine" "mmcv" "mmdet"

New-Item -ItemType Directory -Force -Path ".\configs\swin" | Out-Null
New-Item -ItemType Directory -Force -Path ".\checkpoints" | Out-Null

Write-Host ""
Write-Host "MMDetection is installed. Put the Swin config and checkpoint at:"
Write-Host "  configs\swin\mask_rcnn_swin-t-p4-w7_fpn_3x_coco.py"
Write-Host "  checkpoints\mask_rcnn_swin_tiny_patch4_window7.pth"
Write-Host ""
Write-Host "Those paths are the ones referenced from configs\models.yaml."

if (!$SkipDoctor) {
    & .\venv\Scripts\cv-benchmark.exe doctor --profile transformer_core
}
