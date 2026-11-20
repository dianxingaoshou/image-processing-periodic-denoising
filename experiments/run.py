"""Run the paired study with its original, hash-checked relative paths."""
import argparse
import filecmp
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / ".study"
COMMANDS = {
    "prepare": "prepare_inputs.py",
    "diagnose": "diagnose_phase_refit.py",
    "evaluate": "evaluate_phase_refit.py",
    "benchmark": "benchmark_phase_refit.py",
    "benchmark-controls": "benchmark_phase_controls.py",
    "figures": "plot_phase_refit.py",
}


def study_files():
    # The frozen evaluators record paths relative to the original flat layout.
    # The distributed source tree stores each file once, grouped by purpose.
    for folder in (ROOT / "src", ROOT / "experiments/baselines"):
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".json"}:
                yield path, path.relative_to(folder)
    for path in (ROOT / "experiments").glob("*.py"):
        if path.name != "run.py":
            yield path, Path(path.name)
    for directory in ("data", "results"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and path.suffix in {".json", ".csv", ".md"}:
                yield path, path.relative_to(ROOT)
    yield ROOT / "requirements-lock.txt", Path("requirements-lock.txt")


def stage():
    STUDY.mkdir(exist_ok=True)
    marker = STUDY / ".periodic-denoising-study"
    if not marker.exists():
        if any(STUDY.iterdir()):
            raise ValueError(f"The existing directory {STUDY} contains unrelated files")
        marker.touch()
    seen = set()
    for source, relative in study_files():
        if relative in seen:
            raise ValueError(f"Two sources map to the same study file: {relative}")
        seen.add(relative)
        target = STUDY / relative
        if not target.exists() or not filecmp.cmp(source, target, shallow=False):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["stage", "check", *COMMANDS],
                        help="stage sources, check published hashes, prepare data, or run an experiment")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments passed to the selected experiment")
    args = parser.parse_args()
    stage()
    if args.action == "stage":
        print(STUDY)
        return 0
    if args.action == "check":
        cmd = [sys.executable, "verify_study.py"]
    else:
        cmd = [sys.executable, COMMANDS[args.action]]
    return subprocess.call([*cmd, *args.args], cwd=STUDY)


if __name__ == "__main__":
    raise SystemExit(main())
