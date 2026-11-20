"""Fixed-seed same-photo clean/noisy check on four additional official DTD test1 images."""
import argparse
import csv
import json
import tarfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import numpy as np

from evaluate_weave import projected_residual, read_json, snapshot, write_json
from image_io_v3 import preprocess_image, sha256_file
from project_comparison import AMP, ROOT, array_hash, general_sample
from validation_data_v3 import DTD_IMAGE_BASE, _download


SEED = 20261118
DTD = ROOT/"data/textures_v3"
FROZEN = ROOT/"results/weave/freeze.json"


def plan():
    provenance = read_json(DTD/"provenance.json")
    labels = DTD/"dtd-r1.0.1-labels.tar.gz"
    if sha256_file(labels) != provenance["labels_sha256"]:
        raise ValueError("Unexpected DTD labels archive")
    with tarfile.open(labels, "r:gz") as tar:
        member, = [m for m in tar if m.isfile() and PurePosixPath(m.name).name == "test1.txt"]
        items = tar.extractfile(member).read().decode("utf8").splitlines()
    used = set(read_json(DTD/"selection.json")["selected"])
    rng = np.random.default_rng(SEED)
    chosen = []
    for category in ("knitted", "woven"):
        pool = sorted(s for s in items if s.startswith(category+"/") and s not in used)
        chosen.extend(sorted(str(n) for n in rng.choice(pool, 2, replace=False)))
    return dict(seed=SEED, rule="PCG64; lexical test1 pool; exclude previous 12 filenames; choose "
                "two per category without replacement in order knitted, woven; sort chosen filenames within each category",
                selected=chosen, exclude=sorted(used), labels_sha256=provenance["labels_sha256"],
                inference_freeze_sha256=sha256_file(FROZEN), amplitude=AMP, conditions=["clean","global","chirp"],
                geometry="short-side Lanczos resize and central crop 256; shared float grayscale",
                scope="Four new photographs, two per category. Separate from the 92-input study.")


def evaluate(folder, selection):
    from project_pipeline import load_config
    from project_refined import denoise as previous
    from project_weave import denoise as current

    if snapshot() != read_json(FROZEN):
        raise ValueError("Inference source or configuration changed after selection")
    cfg = load_config()
    sources, rows = [], []
    for filename in selection["selected"]:
        url = DTD_IMAGE_BASE + quote(filename, safe="/")
        src = _download(url, DTD/"source"/filename)
        clean, _ = preprocess_image(src, size=256)
        source = dict(filename=filename, url=url, jpeg_sha256=sha256_file(src), reference_sha256=array_hash(clean))
        sources.append(source)
        ident = "dtd_"+Path(filename).stem
        for kind in selection["conditions"]:
            observed, meta, _ = general_sample(clean, ident, kind, 0. if kind == "clean" else AMP)
            old, old_dec = previous(observed, cfg, return_diagnostics=True)
            new, new_dec = current(observed, cfg, return_diagnostics=True)
            for method, result, decision in (("input",observed,{}), ("previous",old,old_dec), ("current",new,new_dec)):
                if decision and np.any(result[~decision["support_mask"]] != observed[~decision["support_mask"]]):
                    raise AssertionError("Modification outside authorized region")
                gain, off = ("", "") if kind == "clean" else projected_residual(observed-result, observed-clean)
                rows.append(dict(photo=filename, kind=kind, method=method,
                    route=decision.get("route",""), reason=decision.get("reason",""), seed=meta["seed"],
                    mse=float(np.mean((result-clean)**2)), mae=float(np.mean(abs(result-clean))),
                    changed_pixels=int(np.count_nonzero(result != observed)), projection=gain, off_energy=off,
                    observed_sha256=array_hash(observed), output_sha256=array_hash(result)))
    if snapshot() != read_json(FROZEN) or plan() != selection:
        raise ValueError("Frozen inputs or source changed while measuring")
    if len(rows) != 36:
        raise ValueError("Incomplete independent photograph check")
    with (folder/"photos_metrics.csv").open("x", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    write_json(folder/"photos_run.json", dict(sources=sources, rows=36, inputs=12,
               script_sha256=sha256_file(Path(__file__)), selection_sha256=sha256_file(folder/"photos_selection.json"),
               metrics_sha256=sha256_file(folder/"photos_metrics.csv"), freeze_sha256=sha256_file(FROZEN)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["plan","run"])
    p.add_argument("--out", type=Path, default=ROOT/"results/weave_photos_reproduction")
    args = p.parse_args()
    selection = plan()
    selection_path = args.out/"photos_selection.json"
    if args.stage == "plan":
        if selection_path.exists():
            raise FileExistsError(selection_path)
        args.out.mkdir(parents=True, exist_ok=True)
        write_json(selection_path, selection)
        print(selection["selected"])
    else:
        if read_json(selection_path) != selection:
            raise ValueError("Expected a fixed selection made by the plan stage before reading any image")
        if any((args.out/name).exists() for name in ("photos_metrics.csv", "photos_run.json")):
            raise FileExistsError("Use a new results folder")
        evaluate(args.out, selection)
