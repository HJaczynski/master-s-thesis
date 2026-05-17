param(
    [string]$DataDir = "data\coco"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Target = Join-Path $RepoRoot $DataDir
$Downloads = Join-Path $Target "_downloads"
New-Item -ItemType Directory -Force -Path $Downloads | Out-Null

function Download-IfMissing {
    param(
        [string]$Url,
        [string]$OutputPath
    )
    if (Test-Path $OutputPath) {
        Write-Host "Already downloaded: $OutputPath"
        return
    }
    Write-Host "Downloading $Url"
    $ProgressPreference = "SilentlyContinue"
    Invoke-WebRequest -Uri $Url -OutFile $OutputPath
}

function Expand-IfMissing {
    param(
        [string]$Archive,
        [string]$ExpectedPath
    )
    if (Test-Path $ExpectedPath) {
        Write-Host "Already extracted: $ExpectedPath"
        return
    }
    Write-Host "Extracting $Archive"
    Expand-Archive -Path $Archive -DestinationPath $Target -Force
}

$ValZip = Join-Path $Downloads "val2017.zip"
$AnnZip = Join-Path $Downloads "annotations_trainval2017.zip"

Download-IfMissing "http://images.cocodataset.org/zips/val2017.zip" $ValZip
Download-IfMissing "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" $AnnZip

Expand-IfMissing $ValZip (Join-Path $Target "val2017")
Expand-IfMissing $AnnZip (Join-Path $Target "annotations\instances_val2017.json")

Write-Host "COCO validation data is ready under $Target"
