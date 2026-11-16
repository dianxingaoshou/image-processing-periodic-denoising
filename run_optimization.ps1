param([string]$Python = ".\.venv\Scripts\python.exe")
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath $Python)) { throw "Install requirements.txt in .venv first." }
$steps = @(
    @("prepare_optimization_data.py"),
    @("-m", "pytest", "-q", "tests", "original/periodic_notch_starter/test_demo.py", "--junitxml=results/optimization_tests.xml"),
    @("optimization_experiments.py", "calibrate"),
    @("optimization_experiments.py", "audit"),
    @("optimization_experiments.py", "ablations"),
    @("optimization_experiments.py", "summarize"),
    @("optimization_validation.py", "decisions"),
    @("optimization_validation.py", "supplement"),
    @("optimization_validation.py", "verify"),
    @("write_optimization_report.py")
)
foreach ($step in $steps) {
    & $Python @step
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $($step -join ' ')" }
}
