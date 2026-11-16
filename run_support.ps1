param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Output = "results/support_reproduction"
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath $Python)) { throw "Install requirements-lock.txt first." }
& $Python prepare_v3_data.py
if ($LASTEXITCODE -ne 0) { throw "Data preparation failed." }
& $Python -m pytest -q tests original/periodic_notch_starter/test_demo.py --require-bsds "--junitxml=$(Join-Path $Output 'tests.xml')"
if ($LASTEXITCODE -ne 0) { throw "Full tests failed." }
# Reuse committed filenames/widths/parameters; no check-set reselection or tuning.
foreach ($stage in @("development", "freeze", "regression", "check", "report")) {
    & $Python support_experiments.py $stage --out $Output
    if ($LASTEXITCODE -ne 0) { throw "Support study failed: $stage" }
}
& $Python plot_support.py --out $Output --figure (Join-Path $Output "support-ablation.png")
if ($LASTEXITCODE -ne 0) { throw "Plot generation failed." }
& $Python verify_support.py --out $Output
if ($LASTEXITCODE -ne 0) { throw "Artifact verification failed." }
