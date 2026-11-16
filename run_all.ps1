param([string]$Python = ".\.venv\Scripts\python.exe")
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment missing. Run python -m venv .venv and install requirements.txt first."
}
$steps = @(
    @("-m", "pytest", "-q", "tests", "original/periodic_notch_starter/test_demo.py", "--junitxml=results/tests.xml"),
    @("data_pipeline.py", "--root", "data"),
    @("run_experiments.py", "calibrate"),
    @("run_experiments.py", "test"),
    @("analyze_results.py"),
    @("validate_tradeoff.py"),
    @("audit_failures.py"),
    @("verify_artifacts.py")
)
foreach ($step in $steps) {
    & $Python @step
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $($step -join ' ')" }
}
