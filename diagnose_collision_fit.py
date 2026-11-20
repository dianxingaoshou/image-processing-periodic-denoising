"""Offline attribution of fit drift when local legitimate tones coexist with a chirp."""
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares

from check_collision_textures import examples
from evaluate_weave import projected_residual, write_json
from peak_guard import _carrier_geometry
from project_comparison import ROOT
from project_guarded import _restore
from project_peak import denoise as previous
from weak_carrier import _measurements, _verify


def fitted(observed):
    _, diag = previous(observed,return_diagnostics=True)
    rec=diag["evidence"]["guarded_carrier"]
    wave, _ = _carrier_geometry(observed.shape,rec)
    basis4,cells,signal,weights,_ = _measurements(observed)
    coords=np.moveaxis(basis4,-1,0)
    basis=np.stack((coords[0],coords[1],coords[2],coords[0]*coords[1],coords[3]),axis=-1)
    train=(np.indices((4,4)).sum(axis=0).ravel()%2)==0
    change=np.abs(diag["carrier_wave"]-wave) > rec["amplitude"]*1e-7
    clear = np.broadcast_to(~np.any(change,axis=0),observed.shape)
    held=~train[cells]
    training=np.array(rec["training_parameters_bins"])
    _, coef=_verify(training,basis,cells,signal,weights,train)
    train_wave=np.real(coef*np.exp(2j*np.pi*(basis@training)))
    h,w=observed.shape
    final=np.r_[np.asarray(rec["center_frequency"])*[h,w],np.asarray(rec["hessian"])*[h*h,h*w,w*w]]
    record=dict(training_params=training.tolist(),training_amplitude=float(abs(coef)),
                training_phase=float(np.angle(coef)),final_params=final.tolist(),
                final_amplitude=rec["amplitude"],final_phase=rec["final_phase_radians"],
                train_endpoints=int((~held & ((observed==0)|(observed==1))).sum()),
                exterior_pixels=int(clear.sum()),exterior_train_pixels=int((clear&~held).sum()))
    return record,(train_wave,wave),basis,clear,held


def fit_exterior(observed,basis,record,mask,axis_only):
    # Use only pixels from the masked exterior to fit the brightness waveform.
    # Gaussian detrending near a strip boundary can still see its neighbor;
    # the oracle common-exterior evaluation below removes a 15 px buffer.
    brightness=(observed-gaussian_filter(observed,5))[mask]
    active=[0,1,4] if axis_only else list(range(5))
    b=basis[mask][:,active]
    seed=np.array(record["final_params"])[active]
    phase=record["final_phase"]
    amp=record["final_amplitude"]
    initial=np.r_[seed,amp*np.cos(phase),-amp*np.sin(phase)]
    def values(p):
        theta=(2*np.pi)*(b@p[:-2])
        return np.cos(theta),np.sin(theta)
    def objective(p):
        c,s=values(p)
        return p[-2]*c+p[-1]*s-brightness
    def jacobian(p):
        c,s=values(p)
        return np.column_stack(((2*np.pi*(-p[-2]*s+p[-1]*c))[:,None]*b,c,s))
    opt=least_squares(objective, initial, jac=jacobian, bounds=(np.r_[seed-.75,-.25,-.25],
                                                                np.r_[seed+.75,.25,.25]), max_nfev=18)
    p=opt.x
    theta=(2*np.pi)*(basis[...,active]@p[:-2])
    return p[-2]*np.cos(theta)+p[-1]*np.sin(theta),dict(params=p.tolist(),
                  normalized_jacobian_condition=float(np.linalg.cond(opt.jac/np.linalg.norm(opt.jac,axis=0))))


def run():
    selected={(side,name):(ref,obs,inside,base) for side,name,ref,obs,inside,_,base in examples()
              if name in ("stationary_weave_plus_stripe","identical_observation")}
    _,original,_,_ = selected["left","identical_observation"]
    baseline,waves,basis,clear,held=fitted(original)
    cases={}
    for side in ("left","right"):
        ref,observed,inside,base=selected[side,"stationary_weave_plus_stripe"]
        texture=(ref-base)[inside]
        compared,new_waves,_,modified_clear,_=fitted(observed)
        common=(clear & modified_clear).copy()
        # Erode away pixels within 16 columns of a known texture increment of amplitude >0.
        texture_columns=np.any(ref!=base,axis=0)
        halo=np.convolve(texture_columns.astype(float),np.ones(33),mode="same")>0
        common[:,halo]=False
        nodes={}
        def evaluate(name,old_w,new_w):
            a=_restore(original,old_w,1.)
            b=_restore(observed,new_w,1.)
            delta=(b-a)[inside]
            nodes[name]=dict(texture_gain=projected_residual(delta,texture)[0],
                             texture_mse=float(np.mean((delta-texture)**2)),
                             inside_mse=float(np.mean((b[inside]-ref[inside])**2)),
                             inside_wave_drift=float(np.sqrt(np.mean((new_w[inside]-old_w[inside])**2))),
                             whole_stripe_gain=projected_residual(observed-b,observed-ref)[0])
        evaluate("brightness_training",waves[0],new_waves[0])
        evaluate("brightness_calibrated",waves[1],new_waves[1])
        evaluate("oracle_original_fit_only",waves[1],waves[1])
        refit={}
        for name,m1,m2 in (("common_exterior",common,common),("own_exterior",clear,modified_clear)):
            for typ,axial in (("axial",True),("full",False)):
                for half,fitting in (("full",np.ones_like(held)),("training",~held)):
                    a,pa=fit_exterior(original,basis,baseline,m1&fitting,axial)
                    b,pb=fit_exterior(observed,basis,compared,m2&fitting,axial)
                    key=f"{name}_{typ}_{half}"
                    evaluate(key,a,b)
                    refit[key]=dict(original=pa,changed=pb)
        cases[side]=dict(original=baseline,with_texture=compared,common_exterior_pixels=int(common.sum()),
                         common_exterior_training_pixels=int((common&~held).sum()),fits=nodes,exterior_fit_parameters=refit)
    return cases


def report(cases):
    lines=["# 局部真实织纹怎样牵动完整载波拟合", "",
           "在原来的高幅 axis_x_E 观测上，分别给左、右带加入频率接近推算节点、相位有偏移的合法静止纹理。"
           "保留同一真实渐变干扰。测量带内织纹保留时，将含合法纹理的输出减去同干扰、无额外纹理时的输出，"
           "并将差值投影到已知合法织纹增量上。投影比例 1 表示该方向的净增量得到完整保留。", "",
           "| 新增织纹 | 训练截幅点 原 → 加纹 | 训练亮度相位（弧度） | 最终相位（弧度） | 最终振幅 | 可用外部像素 原 → 加纹 |",
           "|---|---:|---:|---:|---:|---:|"]
    for side,r in cases.items():
        a,b=r["original"],r["with_texture"]
        keys=("train_endpoints","training_phase","final_phase","final_amplitude","exterior_pixels")
        fields=[f"{a[k]:.5g} → {b[k]:.5g}" for k in keys]
        lines.append("| "+side+" | "+" | ".join(fields)+" |")
    lines+=["","固定端点恢复算法，替换完整拟合波形。干预列“共用原拟合”离线提供了不含新增织纹时的波形，"
            "目的是检查波形拟合变化能解释多少误差。外部重拟合使用各自观测拟出的非保护列，"
            "估计中央纵横频率、横轴曲率和余弦正弦系数。最小二乘使用去除缓慢亮度变化后的像素。", "",
            "| 区域 | 训练波形完整减除 | 精化波形完整减除 | 共用原拟合 | 仅外部像素轴向重拟 | 外部重拟的真实去除比例 |",
            "|---|---:|---:|---:|---:|---:|"]
    for side,r in cases.items():
        q=r["fits"]
        keys=("brightness_training","brightness_calibrated","oracle_original_fit_only","own_exterior_axial_full")
        values=[q[k]["texture_gain"] for k in keys]+[q["own_exterior_axial_full"]["whole_stripe_gain"]]
        lines.append("| "+side+" | "+" | ".join(f"{v:.4f}" for v in values)+" |")
    lines+=["","两张观测共用同一波形后，额外纹理保留都超过 99%。因此损伤很大一部分来自拟合波形随局部真实"
            "纹理变化。仅靠训练期参数仍受影响。外部重新拟合可改善保留，但图像只提供两段分离的外部条带；"
            "完整外部像素轴向拟合在本例中会过减。外部拟合仅作为独立诊断，不作为复原真值。"
            "JSON 另列按外区交错训练、共用有隔离缓冲的外区、二维曲率模型的各项结果及标准化雅可比条件数。", "",
            "[外区重拟对照与完整参数](fit_drift.json) · [局部织纹测试条件](texture_controls.md)"]
    return "\n".join(lines)+"\n"


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"results/collision83")
    folder=parser.parse_args().out
    folder.mkdir(parents=True,exist_ok=True)
    data=run()
    write_json(folder/"fit_drift.json",data)
    (folder/"fit_drift.md").write_text(report(data),encoding="utf8")
