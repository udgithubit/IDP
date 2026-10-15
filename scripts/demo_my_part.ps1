# One-command demo of the cleaning stage (noise reduction, baseline, threshold) on the Big pieces recording.
#
#   cd "C:\Users\Unmona Das\IDP"
#   .\scripts\demo_my_part.ps1
#
# Optional: .\scripts\demo_my_part.ps1 -Folder "D:\data\Big pieces" -GroupRepo "D:\code\IDP2026_group8"
# Needs: python with numpy, pandas, scipy, matplotlib (the group repo's venv python also works: set -Python).
param(
    [string]$Folder    = "C:\Users\Unmona Das\Documents\IDP\Big pieces reference data (1)",
    [string]$GroupRepo = "C:\Users\Unmona Das\IDP2026_group8",
    [string]$Out       = "$HOME\demo_preprocessing.png",
    [string]$Python    = "python"
)
$ErrorActionPreference = "Stop"

# the file name contains [ ], so -LiteralPath is needed
$csv = (Get-ChildItem -LiteralPath $Folder -Recurse -File | Where-Object { $_.Length -gt 100MB } | Select-Object -First 1).FullName
if (-not $csv) { throw "No recording over 100 MB found in: $Folder" }
Write-Host "Recording: $csv"

Write-Host "`n=== 1. Cleaning stage on its own: 1 kHz, noise reduction, baseline, sigma ==="
& $Python "$PSScriptRoot\demo_preprocessing_standalone.py" "$csv" "$Out"

Write-Host "`n=== 2. The group pipeline on the same recording (cleaning + detection) ==="
Push-Location $GroupRepo
try { & $Python -W ignore tools\run_pipeline.py "$Folder" } finally { Pop-Location }

Write-Host "`nOpening the figure: $Out"
Start-Process $Out
