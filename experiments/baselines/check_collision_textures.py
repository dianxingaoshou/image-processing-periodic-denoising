"""Test protected strips with genuine local texture, using only the historical phase."""
import argparse
from pathlib import Path

import numpy as np

from evaluate_nine83 import DIRECTIONS, samples
from evaluate_weave import projected_residual, read_json, write_json
from image_io_v3 import sha256_file
from project_collision import denoise as current
from project_comparison import ROOT, array_hash
from project_peak import denoise as previous


def examples():
    freeze = read_json(ROOT/"results/nine83/freeze.json")
    spec = next(v for v in freeze["protocol"]["new_variants"] if v["name"] == "axis_x_E")
    case, _ = next((c,a) for c,a in samples(freeze["protocol"],freeze["endpoint_controls"])
                   if c[0]==DIRECTIONS and c[3]=="axis_x_E@0.080")
    image, historical = case[4:6]
    h,w = image.shape
    y,x = np.indices(image.shape,dtype=float)
    columns = x.copy()
    y -= (h-1)/2
    x -= (w-1)/2
    phase = 2*np.pi*(spec["fy"]*y+spec["fx"]*x+spec["dx"]*x*x/(2*(w-1)))
    offset = np.random.default_rng(spec["seed"]).uniform(-np.pi,np.pi)
    wave = .08*np.sin(phase+offset)
    if not np.array_equal(np.clip(image+wave,0,1), historical):
        raise ValueError("First reproduce the historical, fully contaminated observation")
    for side, (lo,hi), node in (("left",(26,123),(28.33,23.20)),("right",(188,256),(28.36,15.78))):
        inside = (columns>=lo) & (columns<hi)
        # Constant inside the designated clean band, eight-column transition outside.
        distance = np.maximum(np.maximum(lo-columns, columns-(hi-1)),0.)
        window = .5+.5*np.cos(np.pi*np.minimum(distance/8.,1.))
        stable_texture = .035*np.cos(2*np.pi*(node[0]*y/h+node[1]*x/w)+offset+np.pi/2)
        ref_texture = np.clip(image+window*stable_texture,0,1)
        for name, ref, obs in (
                ("clean_gap",image,np.clip(image+(1-window)*wave,0,1)),
                ("stationary_weave_gap",ref_texture,np.clip(ref_texture+(1-window)*wave,0,1)),
                ("stationary_weave_plus_stripe",ref_texture,np.clip(ref_texture+wave,0,1)),
                ("identical_observation",np.clip(image+window*wave,0,1),historical)):
            yield side,name,ref,obs,inside,window==0,image


def region(reference, observed, output, mask):
    difference = (output-reference)[mask]
    added = (observed-reference)[mask]
    removal = (observed-output)[mask]
    gain = projected_residual(removal,added)[0] if np.any(added) else None
    return dict(pixels=int(mask.sum()),input_mse=float(np.mean(added**2)),mse=float(np.mean(difference**2)),
                mae=float(np.mean(abs(difference))), gain=gain,
                changed_pixels=int(np.count_nonzero(removal)))


def run():
    rows = []
    paired = {}
    cached = {}
    sources = ("project_collision.py","collision_review.py","check_collision_textures.py")
    hashes = {n:sha256_file(ROOT/n) for n in sources}
    for side, name, reference, obs, inside, exterior, image in examples():
        cases={}
        digest = array_hash(obs)
        for method, restore in (("previous",previous),("current",current)):
            key = method,digest
            if key not in cached:
                cached[key] = restore(obs,return_diagnostics=True)
            out, diag = cached[key]
            review = diag["evidence"].get("collision_review",{})
            cases[method]=dict(route=diag["route"],reason=diag["reason"],review_reason=review.get("reason"),
                               output_sha256=array_hash(out),inside=region(reference,obs,out,inside),
                               exterior=region(reference,obs,out,exterior),
                               whole=region(reference,obs,out,np.ones_like(inside)))
            paired[side,name,method] = out[inside]
        if name=="stationary_weave_plus_stripe":
            paired[side,"texture_increment"] = (reference-image)[inside]
        rows.append(dict(side=side,kind=name,reference_sha256=array_hash(reference),observed_sha256=array_hash(obs),
                         outputs_identical=cases["previous"]["output_sha256"]==cases["current"]["output_sha256"],**cases))
    for row in rows:
        if row["kind"] not in ("stationary_weave_gap","stationary_weave_plus_stripe"):
            continue
        side, name = row["side"], row["kind"]
        base = "clean_gap" if name=="stationary_weave_gap" else "identical_observation"
        texture = paired[side,"texture_increment"]
        for method in ("previous","current"):
            diff = paired[side,name,method]-paired[side,base,method]
            row[method]["paired_stationary_texture_gain"] = projected_residual(diff,texture)[0]
            row[method]["paired_stationary_texture_mse"] = float(np.mean((diff-texture)**2))
    if hashes != {n:sha256_file(ROOT/n) for n in sources}:
        raise RuntimeError("Source changed while running the paired controls")
    return dict(phase="original axis_x_E@0.080 seed",source_sha256=hashes, cases=rows)


def report(data):
    lines=["# 局部无噪区与真实同频纹理的附加检验","","沿用开发照片及旧横向渐变幅值 0.080 的相位。"
           "分别将第 26–122 或 188–255 列定义为测试条带，边界外用 8 列余弦过渡。"
           "测试包括四种情形：仅在条带外加干扰；条带内有额外定频织纹且干扰仅在外部；"
           "全图加干扰并在条带内叠加真实定频织纹；真实织纹与干扰在局部完全一致的不可辨对照。"
           "定频织纹振幅 0.035，以最近推算节点为频率，相位采用旧干扰起始相位加 π/2。"
           "表内误差只统计条带内、不含边界过渡。", "",
           "| 条带与条件 | 带内输入 MSE | 带内 MSE 上版 → 本版 | 带内变化像素 上版 → 本版 | 整图去除比例 上版 → 本版 | 整图输出逐值相同 | 本版决定 |",
           "|---|---:|---:|---:|---:|---|---|"]
    for r in data["cases"]:
        o,n=r["previous"],r["current"]
        p,q=o["inside"],n["inside"]
        decision=n["review_reason"] or n["reason"]
        lines.append(f"| {r['side']} / {r['kind']} | {p['input_mse']:.6g} | {p['mse']:.6g} → {q['mse']:.6g} | "
                     f"{p['changed_pixels']} → {q['changed_pixels']} | {o['whole']['gain']:.4f} → {n['whole']['gain']:.4f} | "
                     f"{'是' if r['outputs_identical'] else '否'} | {decision} |")
    lines += ["","再用带合法静止织纹的恢复图减去同一干扰条件下未加入该织纹时的恢复图，"
              "把差值投影到已知合法织纹新增量上。比例 1 表示保留该方向的全部变化。下表另列差值与真实织纹新增量的"
              "带内 MSE，以计入非平行误差。这可以区分总残留变少和真实织纹的保留。","",
              "| 条带与条件 | 合法织纹增量投影比例 上版 → 本版 | 增量误差 MSE 上版 → 本版 |",
              "|---|---:|---:|"]
    for r in data["cases"]:
        o,n = r["previous"],r["current"]
        if "paired_stationary_texture_gain" in o:
            lines.append(f"| {r['side']} / {r['kind']} | {o['paired_stationary_texture_gain']:.4f} → "
                         f"{n['paired_stationary_texture_gain']:.4f} | {o['paired_stationary_texture_mse']:.6g} → "
                         f"{n['paired_stationary_texture_mse']:.6g} |")
    lines += ["","最后一种情形逐像素沿用原高幅观测，只更换清晰参考，将条带内同一相位、幅度、空间变化的波形视作"
              "本来就有的内容。任何单图确定性算法都会给同一输入输出相同恢复图。它测量单图无法区分两种成因时可能"
              "发生的局部损伤。前三种的观测确实重新生成，可分别检验拒绝路径和有真实同频织纹时的结果。", "",
              "仅表示这张照片及本次局部构造。干净条带输出有变化为误改；两版输出相同只表明没有新增误改。"
              "[完整像素误差、外侧条纹减除比例及决策](texture_controls.json)","","复现命令为 "
              "`python check_collision_textures.py --out results/collision83_reproduction`。"]
    return "\n".join(lines)+"\n"


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"results/collision83")
    folder=parser.parse_args().out
    folder.mkdir(parents=True,exist_ok=True)
    data=run()
    write_json(folder/"texture_controls.json",data)
    (folder/"texture_controls.md").write_text(report(data),encoding="utf8")
