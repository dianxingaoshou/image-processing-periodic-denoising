"""Evaluate real local weave and clean-gap controls with paired input and output pixels."""
import numpy as np

from check_collision_textures import examples
from evaluate_narrow83 import check_output
from evaluate_weave import projected_residual, read_json
from project_comparison import ROOT, array_hash


def record(group, variant, side, kind, ref, obs, inside, restorations):
    row = dict(group=group,variant=variant,side=side,kind=kind,
               reference_sha256=array_hash(ref), observed_sha256=array_hash(obs))
    increment = obs-ref
    for method, (out,diag) in restorations.items():
        check_output(obs,out,diag)
        removal = obs-out
        row.update({method+"_sha256":array_hash(out), method+"_reason":diag["reason"],
                    method+"_whole_gain":projected_residual(removal,increment)[0],
                    method+"_inside_mse":float(np.mean((out[inside]-ref[inside])**2)),
                    method+"_inside_changes":int(np.count_nonzero(removal[inside])),
                    method+"_texture_gain":"", method+"_texture_mse":""})
    return row


def marginal(row, method, after, before, texture):
    difference = after-before
    row[method+"_texture_gain"] = projected_residual(difference, texture)[0]
    row[method+"_texture_mse"] = float(np.mean((difference-texture)**2))


def controls(plan, target_outputs, previous, current):
    methods = dict(previous=previous, current=current)
    archive = {(r["side"],r["kind"]):r for r in read_json(ROOT/"results/collision83/texture_controls.json")["cases"]}
    rows = []
    alternative_refs = []
    cache, paired, increments = {}, {}, {}
    for side, kind, ref, obs, inside, _, natural in examples():
        digest = array_hash(obs)
        if digest not in cache:
            cache[digest] = {m:func(obs,return_diagnostics=True) for m,func in methods.items()}
        result = cache[digest]
        row = record("historical","axis_x_E@0.080",side,kind,ref,obs,inside,result)
        saved = archive[side,kind]
        if (row["previous_sha256"],row["reference_sha256"],digest) != (
                saved["current"]["output_sha256"],saved["reference_sha256"],saved["observed_sha256"]):
            raise ValueError("The historical local texture control changed")
        rows.append(row)
        for m,(image,_) in result.items():
            paired[side,kind,m] = image[inside]
        if kind=="stationary_weave_plus_stripe":
            increments[side] = (ref-natural)[inside]
    for r in rows:
        side, kind = r["side"],r["kind"]
        if kind in ("stationary_weave_gap","stationary_weave_plus_stripe"):
            base = "clean_gap" if kind=="stationary_weave_gap" else "identical_observation"
            for m in methods:
                marginal(r,m,paired[side,kind,m],paired[side,base,m],increments[side])
    spec = next(v for v in read_json(ROOT/"results/nine83/protocol.json")["new_variants"] if v["name"]=="axis_x_E")
    for seed in plan["target_seeds"]:
        offset = np.random.default_rng(seed).uniform(-np.pi,np.pi)
        for amp in plan["amplitudes"]:
            variant = f"axis_x_E_s{seed}@{amp:.3f}"
            clean, observed, old_plain, new_plain = target_outputs[variant]
            h,w = clean.shape
            yy,xx = np.indices(clean.shape,dtype=float)
            y,x = yy-(h-1)/2, xx-(w-1)/2
            wave = amp*np.sin(2*np.pi*(spec["fy"]*y+spec["fx"]*x+spec["dx"]*x*x/(2*(w-1)))+offset)
            if not np.array_equal(np.clip(clean+wave,0,1),observed):
                raise ValueError(f"Texture challenge did not reproduce original {variant}")
            for side,lo,hi,node in (("left",26,123,(28.33,23.20)),("right",188,256,(28.36,15.78))):
                inside = (xx>=lo)&(xx<hi)
                dist = np.maximum(np.maximum(lo-xx,xx-(hi-1)),0.)
                window = .5+.5*np.cos(np.pi*np.minimum(dist/8.,1.))
                local = .035*np.cos(2*np.pi*(node[0]*y/h+node[1]*x/w)+offset+np.pi/2)
                textured = np.clip(clean+window*local,0,1)
                # Sensitivity analysis: same digital observation and outputs, with the
                # local chirp attributed to the photograph in the alternative reference.
                alt = np.clip(clean+window*wave,0,1)
                if not np.array_equal(alt[inside],observed[inside]):
                    raise ValueError("Ambiguous reference must be clean inside the designated band")
                alternative_refs.append(dict(variant=variant,side=side, observed_sha256=array_hash(observed),
                    alternative_reference_sha256=array_hash(alt),input_inside_mse=0.,
                    previous_inside_mse=float(np.mean((old_plain[inside]-alt[inside])**2)),
                    current_inside_mse=float(np.mean((new_plain[inside]-alt[inside])**2))))
                cases = [("stationary_weave_plus_stripe",textured,np.clip(textured+wave,0,1))]
                if amp==max(plan["amplitudes"]):
                    outside = (1-window)*wave
                    cases.extend([("clean_gap",clean,np.clip(clean+outside,0,1)),
                                  ("stationary_weave_gap",textured,np.clip(textured+outside,0,1))])
                baseline = dict(previous=old_plain[inside], current=new_plain[inside])
                for kind,ref,obs in cases:
                    result = {m:f(obs,return_diagnostics=True) for m,f in methods.items()}
                    r = record("target",variant,side,kind,ref,obs,inside,result)
                    for m,(out,_) in result.items():
                        if kind=="clean_gap":
                            baseline[m] = out[inside]
                        else:
                            marginal(r,m,out[inside],baseline[m],(textured-clean)[inside])
                    rows.append(r)
    if len(rows)!=24 or len(alternative_refs)!=8:
        raise ValueError("Missing local texture controls")
    return rows, alternative_refs
