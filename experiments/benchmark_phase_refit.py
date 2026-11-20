"""Balanced paired end-to-end API timings on the two problem phases and unseen phases."""
import argparse
import csv
import os
from pathlib import Path
import platform
from statistics import median
from time import perf_counter_ns

import numpy as np
import scipy

from evaluate_narrow83 import write_csv
from evaluate_phase_refit import PLAN, samples
from evaluate_weave import read_json, write_json
from image_io_v3 import sha256_file
from project_collision import denoise as previous
from project_comparison import ROOT, array_hash


def benchmark(folder, rounds=4):
    from project_phase import denoise as current
    if rounds<4 or rounds%2:
        raise ValueError("Select an even number of at least four AB/BA rounds")
    archive=read_json(folder/"run.json")
    if archive["artifact_sha256"]["metrics.csv"]!=sha256_file(folder/"metrics.csv"):
        raise ValueError("Changed formal metrics")
    with (folder/"metrics.csv").open(encoding="utf8",newline="") as f:
        records={(r["cohort"],r["image_id"],r["kind"],r["variant"],r["method"]):r for r in csv.DictReader(f)}
    plan=read_json(PLAN)
    cases=[]
    for c,_ in samples(plan):
        cohort,image_id,kind,variant,_,obs,_=c
        key=cohort,image_id,kind,str(variant)
        reference=records[*key,"current"]
        group=reference["group"]
        if key==("archived_photo",plan["target"],"clean",""):
            group="clean"
        elif image_id=="69022" and kind in ("edge8","edge16") and str(variant)=="3":
            group="narrow_edges"
        elif group not in ("target","new_phases"):
            continue
        if array_hash(obs)!=reference["observed_sha256"]:
            raise ValueError("Timed input differs from the evaluated input")
        cases.append((group,key,obs))
    if len(cases)!=13:
        raise ValueError("Incomplete benchmark cohort")
    methods=(("previous",previous),("current",current))
    rng=np.random.default_rng(704239)
    timings=[]
    for rep in range(-1,rounds):
        for i in rng.permutation(len(cases)):
            group,key,obs=cases[i]
            for position,(name,method) in enumerate(methods if (rep+i)%2==0 else methods[::-1],1):
                start=perf_counter_ns()
                out=method(obs)
                ms=(perf_counter_ns()-start)/1e6
                if array_hash(out)!=records[*key,name]["output_sha256"]:
                    raise ValueError(f"Ordinary entry disagrees with evaluation for {key}/{name}")
                if rep>=0:
                    timings.append(dict(round=rep+1, group=group, cohort=key[0], image_id=key[1], kind=key[2],
                                        variant=key[3], method=name, order=position, ms=ms))
        print(f"Timing round {rep+1}/{rounds}",flush=True)
    stats={}
    def summarize(rows):
        old,new=([sum(r["ms"] for r in rows if r["round"]==i+1 and r["method"]==m)
                  for i in range(rounds)] for m,_ in methods)
        ratios=np.divide(new,old).tolist()
        return dict(previous_median_ms=median(old), current_median_ms=median(new),
                    paired_median_ratio=median(ratios), paired_range=[min(ratios), max(ratios)])
    for group in ("all","target","new_phases","clean","narrow_edges"):
        stats[group]=summarize([r for r in timings if group=="all" or r["group"]==group])
    stats["core7"]=summarize([r for r in timings if r["group"]!="new_phases"])
    for _,key,_ in cases:
        stats[" / ".join(key)]=summarize([r for r in timings if (r["cohort"],r["image_id"],r["kind"],r["variant"])==key])
    write_csv(folder/"timing.csv",timings)
    write_json(folder/"timing.json",dict(inputs=len(cases),rounds=rounds,methods=[m for m,_ in methods],
               runtime=dict(cpu=os.environ.get("PROCESSOR_IDENTIFIER"),python=platform.python_version(),
                            numpy=np.__version__,scipy=scipy.__version__),
               scope="Complete API, excluding synthesis, I/O, hashing and scoring. One warm-up per method/input. Balanced serial AB/BA.",
               groups=stats,metrics_sha256=archive["artifact_sha256"]["metrics.csv"],
               timing_sha256=sha256_file(folder/"timing.csv"), source_sha256=sha256_file(Path(__file__).resolve())))
    print("all",stats["all"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=ROOT/"results/phase_refit")
    p.add_argument("--rounds",type=int,default=4)
    args=p.parse_args()
    benchmark(args.out,args.rounds)
