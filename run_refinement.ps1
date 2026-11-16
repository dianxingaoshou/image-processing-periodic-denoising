param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Output = "results/refinement_reproduction"
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath $Python)) { throw "Install requirements-lock.txt first." }
& $Python prepare_v3_data.py
if ($LASTEXITCODE -ne 0) { throw "Data preparation failed." }
& $Python -m pytest -q tests original/periodic_notch_starter/test_demo.py --require-bsds
if ($LASTEXITCODE -ne 0) { throw "Full tests failed." }
# Reuse committed IDs and paired recipes; never reselect confirmation images.
foreach ($stage in @("development", "freeze", "regression", "check", "report")) {
    & $Python refinement_experiments.py $stage --out $Output
    if ($LASTEXITCODE -ne 0) { throw "Focused study failed: $stage" }
}
& $Python plot_refinement.py --out $Output --figure (Join-Path $Output "risk-summary.png")
if ($LASTEXITCODE -ne 0) { throw "Plot generation failed." }
