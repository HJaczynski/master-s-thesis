param(
    [ValidateSet("smoke", "overnight", "yolo26_full", "throughput")]
    [string]$Profile = "overnight",
    [switch]$DryRun,
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Runner = ".\venv\Scripts\cv-benchmark.exe"
if (!(Test-Path $Runner)) {
    .\venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
}

& $Runner doctor --profile $Profile

$Args = @("run-nightly", "--profile", $Profile)
if ($DryRun) {
    $Args += "--dry-run"
}
if ($Limit -gt 0) {
    $Args += @("--limit", "$Limit")
}

& $Runner @Args
