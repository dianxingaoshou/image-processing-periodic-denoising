"""Check the released source and result fingerprints without loading images."""
import csv
import json
from hashlib import sha256
from pathlib import Path

from evaluate_phase_refit import snapshot


def verify():
    results = Path("results/phase_refit")
    frozen = json.loads((results / "freeze.json").read_text(encoding="utf8"))
    if snapshot() != frozen:
        raise ValueError("The study source, baseline, or protocol differs from the published freeze")
    run = json.loads((results / "run.json").read_text(encoding="utf8"))
    for name, expected in run["artifact_sha256"].items():
        if sha256((results / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Changed study result: {name}")
    timing = json.loads((results / "timing.json").read_text(encoding="utf8"))
    files = [(results / "timing.csv", "timing_sha256"), (results / "metrics.csv", "metrics_sha256"),
             (Path("benchmark_phase_refit.py"), "source_sha256")]
    for path, key in files:
        if sha256(path.read_bytes()).hexdigest() != timing[key]:
            raise ValueError(f"Changed timing input or output: {path}")
    controls = json.loads((results / "timing_controls.json").read_text(encoding="utf8"))
    if (controls["formal_texture_sha256"] != run["artifact_sha256"]["texture_controls.csv"] or
            controls["source_sha256"] != sha256(Path("benchmark_phase_controls.py").read_bytes()).hexdigest()):
        raise ValueError("The texture-control benchmark or its scored inputs changed")
    with (results / "metrics.csv").open(encoding="utf8", newline="") as stream:
        count = sum(1 for _ in csv.DictReader(stream))
    print(f"Verified {len(frozen['source_sha256'])} study dependencies and {count} image/method records.")


if __name__ == "__main__":
    verify()
