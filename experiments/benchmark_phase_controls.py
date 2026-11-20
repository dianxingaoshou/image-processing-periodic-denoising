"""Supplemental timing of one newly accepted unseeded texture case and two controls."""
import argparse
import csv
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np

from evaluate_weave import read_json, write_json
from image_io_v3 import sha256_file
from project_collision import denoise as previous
from project_comparison import ROOT, array_hash
from project_phase import denoise as current


def benchmark(folder):
    variant = "axis_x_E_s741209@0.080"
    plan = read_json(ROOT/"results/nine83/protocol.json")
    spec = next(v for v in plan["new_variants"] if v["name"]=="axis_x_E")
    run = read_json(folder/"run.json")
    if sha256_file(folder/"texture_controls.csv") != run["artifact_sha256"]["texture_controls.csv"]:
        raise ValueError("The formal texture controls changed")
    with (folder/"texture_controls.csv").open(encoding="utf8",newline="") as f:
        known = {(r["side"],r["kind"]):r for r in csv.DictReader(f) if r["variant"]==variant}
    with np.load(folder/"examples"/(variant+".npz")) as data:
        clean = data["clean"]
    h,w = clean.shape
    yy,xx = np.indices(clean.shape,dtype=float)
    y,x = yy-(h-1)/2,xx-(w-1)/2
    offset = np.random.default_rng(741209).uniform(-np.pi,np.pi)
    wave = .08*np.sin(2*np.pi*(spec["fy"]*y+spec["fx"]*x+spec["dx"]*x*x/(2*(w-1)))+offset)
    cases = []
    for side,lo,hi,node in (("left",26,123,(28.33,23.20)),("right",188,256,(28.36,15.78))):
        distance = np.maximum(np.maximum(lo-xx,xx-(hi-1)),0.)
        window = .5+.5*np.cos(np.pi*np.minimum(distance/8.,1.))
        local = .035*np.cos(2*np.pi*(node[0]*y/h+node[1]*x/w)+offset+np.pi/2)
        textured = np.clip(clean+window*local,0,1)
        cases.append((side,"stationary_weave_plus_stripe",np.clip(textured+wave,0,1)))
        if side=="left":
            cases.append((side,"clean_gap",np.clip(clean+(1-window)*wave,0,1)))
    methods = (("previous",previous),("current",current))
    results = []
    for side,kind,obs in cases:
        ref = known[side,kind]
        if array_hash(obs)!=ref["observed_sha256"]:
            raise ValueError(f"Supplemental case differs from formal input {side}/{kind}")
        stats = dict(previous=[],current=[])
        for rep in range(-1,4):
            for name,f in (methods if rep%2==0 else methods[::-1]):
                start = perf_counter_ns()
                out = f(obs)
                ms = (perf_counter_ns()-start)/1e6
                if array_hash(out)!=ref[name+"_sha256"]:
                    raise ValueError(f"Supplemental output changed {side}/{kind}/{name}")
                if rep>=0:
                    stats[name].append(ms)
        ratios = np.divide(stats["current"],stats["previous"]).tolist()
        results.append(dict(side=side,kind=kind,observed_sha256=ref["observed_sha256"],
                            previous_sha256=ref["previous_sha256"],current_sha256=ref["current_sha256"],
                            previous_reason=ref["previous_reason"],current_reason=ref["current_reason"],
                            ms=stats, median_ms={name:median(values) for name,values in stats.items()},
                            paired_median_ratio=median(ratios),paired_range=[min(ratios),max(ratios)]))
        print(side,kind,results[-1]["median_ms"],flush=True)
    write_json(folder/"timing_controls.json",dict(variant=variant, rounds=4,
               scope="Supplemental post-hoc timing; one warmup each. Complete APIs in serial AB/BA order. Excludes I/O, hashing and synthesis. Not pooled into the core benchmark.",
               results=results,source_sha256=sha256_file(Path(__file__).resolve()),
               formal_texture_sha256=run["artifact_sha256"]["texture_controls.csv"]))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=ROOT/"results/phase_refit")
    benchmark(p.parse_args().out)
