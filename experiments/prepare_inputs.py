"""Prepare only the source photographs and reference arrays used by the released study."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

import data_pipeline
from evaluate_project_refinement import EXTRA_DEV
from image_io_v3 import preprocess_image, sha256_file
from project_comparison import DEV_IDS
from validation_data_v3 import _download


DATA = Path("data")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verified(path, expected):
    if sha256_file(path) != expected:
        raise ValueError(f"Unexpected hash for {path}")


def reuse(cache, rel, expected):
    target = DATA / rel
    if not target.is_file() and cache and (cache / rel).is_file():
        verified(cache / rel, expected)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cache / rel, target)
    if not target.is_file():
        return False
    verified(target, expected)
    return True


def prepare(cache):
    manifest = read(DATA / "v3_manifest.json")
    dev = [r for r in manifest["images"] if r["id"] in {*DEV_IDS, *EXTRA_DEV}]
    protocol = read(Path("rejection_protocol.json"))
    edge = next(r for r in protocol["regression_images"] if r["id"] == "69022")
    bsds = [*dev, edge]
    missing = [r for r in bsds if not reuse(cache, r["source_jpg"], r["source_sha256"])]
    if missing:
        archive_rel = Path("source/BSR_bsds500.tgz")
        digest = read(DATA / "manifest.json")["archive_sha256"]
        archive = DATA / archive_rel
        if not reuse(cache, archive_rel, digest):
            data_pipeline._download(archive)
            verified(archive, digest)
        data_pipeline._extract_images(archive, DATA / "source/images")
    for r in bsds:
        verified(DATA / r["source_jpg"], r["source_sha256"])
    for r in manifest["textures"]:
        if not reuse(cache, r["source_jpg"], r["source_sha256"]):
            verified(_download(r["source_url"], DATA / r["source_jpg"]), r["source_sha256"])
    for r in [*dev, *manifest["textures"]]:
        if not reuse(cache, r["npy_path"], r["clean_sha256"]):
            pixels, _ = preprocess_image(DATA / r["source_jpg"], 256)
            target = DATA / r["npy_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, pixels)
            verified(target, r["clean_sha256"])
    extra = read(Path("results/weave/photos_run.json"))["sources"]
    for r in extra:
        rel = Path("textures_v3/source") / r["filename"]
        if not reuse(cache, rel, r["jpeg_sha256"]):
            verified(_download(r["url"], DATA / rel), r["jpeg_sha256"])
    print(f"Verified {len(bsds)} BSDS photographs and {len(manifest['textures']) + len(extra)} DTD photographs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, help="optional absolute path of an existing prepared data directory")
    args = parser.parse_args()
    prepare(args.cache)
