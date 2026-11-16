param([string]$Python = ".\.venv\Scripts\python.exe")
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath $Python)) { throw "Install requirements-lock.txt in the matching Python runtime first." }
$steps = @(
    @("prepare_v3_data.py"),
    @("-m", "pytest", "-q", "tests", "original/periodic_notch_starter/test_demo.py", "--require-bsds", "--junitxml=results/v3/tests.xml"),
    @("v3_experiments.py", "calibrate"),
    @("v3_experiments.py", "audit"),
    @("v3_experiments.py", "stress"),
    @("v3_experiments.py", "textures"),
    @("v3_experiments.py", "real"),
    @("v3_evidence.py", "matched"),
    @("v3_evidence.py", "boundaries"),
    @("texture_tradeoff_v3.py"),
    @("v3_experiments.py", "summarize"),
    @("v3_failure_audit.py"),
    @("v3_verify.py"),
    @("v3_report.py")
)
foreach ($step in $steps) {
    & $Python @step
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $($step -join ' ')" }
}
