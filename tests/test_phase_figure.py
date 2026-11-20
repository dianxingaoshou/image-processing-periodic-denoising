"""Rebuild public examples from the registered inputs and keep the cached fast path."""
import csv
import json

import numpy as np

import evaluate_phase_refit
from image_io_v3 import sha256_file
from plot_phase_refit import read_cases
from project_comparison import array_hash
import project_collision
import project_phase


def test_plot_rebuilds_only_missing_displayed_inputs(tmp_path, monkeypatch):
    plan = dict(target="woven", target_seeds=[1], new_phase_seeds=[2], amplitudes=[0.045])
    arrays = dict(clean=np.zeros((4, 5)), input=np.full((4, 5), .2),
                  previous=np.full((4, 5), .1), current=np.full((4, 5), .03))
    names = ["axis_x_E_s1@0.045", "axis_x_E_s2@0.045"]
    examples = tmp_path / "examples"
    examples.mkdir()
    np.savez(examples / (names[0]+".npz"), **arrays)
    fields = "variant method image_id reference_sha256 observed_sha256 output_sha256 noise_gain".split()
    with (tmp_path / "metrics.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for name in names:
            for method in ("input", "previous", "current"):
                writer.writerow((name, method, "woven", array_hash(arrays["clean"]),
                                 array_hash(arrays["input"]), array_hash(arrays[method]), .4))
    (tmp_path / "run.json").write_text(json.dumps(dict(artifact_sha256={
        "metrics.csv": sha256_file(tmp_path/"metrics.csv")})), encoding="utf8")
    checked = []
    def method(label):
        def denoise(obs, cfg):
            checked.append(label)
            return arrays[label]
        return denoise
    monkeypatch.setattr(project_collision, "denoise", method("previous"))
    monkeypatch.setattr(project_phase, "denoise", method("current"))
    cases = [(("unused", "woven", "", n, arrays["clean"], arrays["input"], None), .045) for n in names]
    monkeypatch.setattr(evaluate_phase_refit, "samples", lambda unused: iter(cases))
    built = read_cases(tmp_path, plan)
    assert checked == ["previous", "current"] and len(built) == 2
    np.testing.assert_array_equal(built[2, .045][0]["current"], arrays["current"])
    np.savez(examples / (names[1]+".npz"), **arrays)
    read_cases(tmp_path, plan)
    assert checked == ["previous", "current"]
