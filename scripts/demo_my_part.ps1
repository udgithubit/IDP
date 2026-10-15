# Two-part demo for the cleaning stage.
#   Part 1: the noise level and the threshold on one recording (Big pieces)
#   Part 2: the analysis over all the recordings (group pipeline) and a summary table
#
#   cd "C:\Users\Unmona Das\IDP"
#   .\scripts\demo_my_part.ps1
#
# Run it once before the meeting so the group pipeline's cache is warm (the first read of each CSV takes a few seconds).
# Needs: python with numpy, pandas, scipy, matplotlib (use -Python for the group repo's venv python if needed).
param(
    [string]$Base      = "C:\Users\Unmona Das\Documents\IDP",
    [string]$GroupRepo = "C:\Users\Unmona Das\IDP2026_group8",
    [string]$Out       = "$HOME\demo_preprocessing.png",
    [string]$PiecesDir = "$HOME\pipeline_out_demo",
    [string]$Python    = "python"
)
$ErrorActionPreference = "Stop"
$folders = @("$Base\Big pieces reference data (1)",
             "$Base\Data for suction of known volumes + 4½ h hours [2025] (1)",
             "$Base\Tumor 1 gr - 2 gr - 3 gr with resuction",
             "$Base\Measurements from demo sessions")

function Get-Recordings($dirs) {
    # recordings are the big files; some have no .csv extension, so list files instead of using *.csv
    $dirs | ForEach-Object { Get-ChildItem -LiteralPath $_ -Recurse -File } |
        Where-Object { $_.Length -gt 1MB -and $_.Name -notmatch "Copy" -and $_.Extension -notin ".7z", ".zip", ".png", ".odt", ".json", ".txt" } |
        ForEach-Object { $_.FullName }
}

# ---------------------------------------------------------------- Part 1
$csv = (Get-ChildItem -LiteralPath $folders[0] -Recurse -File | Where-Object { $_.Length -gt 100MB } | Select-Object -First 1).FullName
if (-not $csv) { throw "No recording over 100 MB found in: $($folders[0])" }
Write-Host "`n=== PART 1: noise level and threshold on one recording ===" -ForegroundColor Cyan
Write-Host "Recording: $csv"
& $Python "$PSScriptRoot\demo_preprocessing_standalone.py" "$csv" "$Out"
Write-Host "`nFigure saved: $Out   (open it with:  start `"$Out`")"
Read-Host "`nPress Enter to continue to Part 2"

# ---------------------------------------------------------------- Part 2
$files = @(Get-Recordings $folders)
Write-Host "`n=== PART 2: the analysis over all the recordings ($($files.Count) files) ===" -ForegroundColor Cyan
Remove-Item -Recurse -Force $PiecesDir -ErrorAction SilentlyContinue
$sw = [Diagnostics.Stopwatch]::StartNew()
Push-Location $GroupRepo
try { & $Python -W ignore tools\run_pipeline.py @files --out $PiecesDir } finally { Pop-Location }
$sw.Stop()
Write-Host ("`n{0} recordings analysed in {1:N1} s (recordings opened before are read from the cache)" -f $files.Count, $sw.Elapsed.TotalSeconds) -ForegroundColor Green

$summary = Join-Path $PSScriptRoot "..\proposals\summarize_results.py"
if (Test-Path $summary) {
    Write-Host "`n--- summary: pieces per recording, volume calibration, tumor trend ---"
    & $Python $summary $PiecesDir
}
