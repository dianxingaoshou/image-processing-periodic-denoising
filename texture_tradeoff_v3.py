"""Post-hoc exploratory DTD paired-noise/clean-photo trade-off diagnostic.

This standalone supplement was requested after v3 was frozen. It neither edits
the frozen source inventory nor selects a new alpha. The horizontal loss is
whole-image restoration MSE against the original natural photo after added
noise; the vertical loss is the same mode/alpha's modification MSE on that
photo without added noise. The field name contaminated_mse is retained solely
to reuse the previously frozen finite-grid frontier/match implementation.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

from denoiser_v3 import prepare,restore
from evaluation_v3 import tail_mean
from run_experiments import load_clean
from v3_experiments import DATA,OUT,ROOT,CONFIG,load_frozen,records
from v3_evidence import frontier,match
from validation_data_v3 import make_sample

STRENGTHS=(0.,.25,.5,.75,1.,1.25,1.5,2.)
MODES=('coupled','decoupled','unprotected')
KINDS=('global','chirp')
FOLDER=OUT/'texture_tradeoff'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a,dtype='<f8').tobytes()).hexdigest()


def dump(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def write_csv(path,rows,fields=None):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or list(rows[0]));writer.writeheader();writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def output(prepared,mode,alpha):
    return restore(prepared,strength=alpha,combination='coupled' if mode=='coupled' else 'decoupled',
                   conservative=mode!='unprotected')


def plots(means,matched,coverage,ids):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for row,kind in enumerate(KINDS):
        ax=axs[row,0]
        for mode in MODES:
            points=[r for r in means if r['kind']==kind and r['mode']==mode]
            ax.plot([1-r['contaminated_mse']/r['input_contaminated_mse'] for r in points],
                    [r['clean_mse'] for r in points],'-o',label=mode,markersize=4)
        ax.set_title(kind+' | actual fixed-grid points');ax.legend(fontsize=8)
        ax.set_xlabel('Noisy-pair whole-image restoration\nMSE reduction fraction')
        ax.set_ylabel('Original-photo whole-image\nmodification MSE (no added noise)')
        for ax,pair in zip(axs[row,1:],('coupled','unprotected')):
            points=[r for r in matched if r['kind']==kind and r['group']=='subset_mean' and r['comparison']=='decoupled_vs_'+pair]
            if points:
                x=[r['matched_contaminated_mse_reduction_fraction'] for r in points]
                ax.plot(x,[r['decoupled_clean_mse'] for r in points],label='decoupled')
                ax.plot(x,[r[pair+'_clean_mse'] for r in points],label=pair);ax.legend(fontsize=8)
            else:ax.text(.5,.5,'No positive common removal interval',transform=ax.transAxes,ha='center',va='center',fontsize=9)
            ax.set_title(kind+' | decoupled vs '+pair)
            ax.set_xlabel('Matched noisy-pair whole-image\nrestoration MSE reduction fraction')
            ax.set_ylabel('Original-photo whole-image\nmodification MSE (no added noise)')
        for ax in axs[row]:ax.grid(alpha=.2);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    fig.suptitle('DTD12 exploratory paired diagnostic | no parameter selection',fontsize=13)
    fig.savefig(FOLDER/'paired_tradeoff.png',dpi=160);plt.close(fig)
    matrix=np.array([[next(r['matched_points']>0 for r in coverage if r['image_id']==image_id and r['kind']==kind and r['comparison']=='decoupled_vs_'+pair)
        for kind in KINDS for pair in ('coupled','unprotected')] for image_id in ids],dtype=int)
    fig,ax=plt.subplots(figsize=(9,6.5),layout='constrained');ax.imshow(matrix,cmap='Blues',vmin=0,vmax=1,aspect='auto')
    ax.set_xticks(range(4),['global vs coupled','global vs unprotected','chirp vs coupled','chirp vs unprotected'],rotation=18,ha='right')
    ax.set_yticks(range(len(ids)),ids,fontsize=8)
    for y in range(len(ids)):
        for x in range(4):ax.text(x,y,'matched' if matrix[y,x] else 'no interval',ha='center',va='center',fontsize=8,color='white' if matrix[y,x] else 'black')
    ax.set_title('All12 photos and both added-noise conditions retained | exploratory')
    fig.savefig(FOLDER/'individual_coverage.png',dpi=160);plt.close(fig)


def note(operating,coverage,matched,config_hash,alpha,checks):
    lines=['# DTD配对去噪与纹理损伤的补充探索性诊断','',
        '本诊断在v3冻结后新增，未用于调参，不构成新的独立确认性主实验。没有修改冻结算法、评分公式、配置或既有报告；不从曲线选择新的alpha。','',
        '固定使用原12张DTD官方test1照片。每张分别运行未新增干扰、全局定频0.12和频率渐变0.12，共36个输入；每个输入只估计一次候选残差，再扫描3模式和8个既定强度。',
        '图中横轴是“含噪配对图的全图恢复MSE降低”，纵轴是“同源未新增干扰照片的全图修改MSE”。这是两次不同输入的paired-control比较。'
        '横轴损失为加噪照片恢复结果相对原照片的全图MSE；纵轴损伤为同一照片未新增干扰、相同模式及alpha下的修改MSE。'
        'CSV保留contaminated_mse字段以复用冻结匹配代码，但这里它明确表示全图恢复MSE，绝不是局部含噪区域误差。','',
        'DTD原照片不是严格物理无噪声图；这里的原图仅作为自行添加干扰的配对参考。照片中的合法纹理也可能被误删，因此不能只靠clean照片修改变少来宣称保护成功。','',
        f'冻结配置SHA256：`{config_hash}`。原冻结alpha={alpha:g}；以下运行点沿用该值，不选择新值。','',
        '## 冻结运行点','',
        '| 加噪条件 | 模式 | 恢复MSE均值 | 输入MSE均值 | 原照片修改MSE均值 | 原照片修改MAE均值 | 原照片最坏20% MAE均值 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in operating:
        lines.append(f"| {r['kind']} | {r['mode']} | {r['mean_noisy_restoration_mse']:.6g} | {r['mean_input_noisy_mse']:.6g} | {r['mean_clean_modification_mse']:.6g} | {r['mean_clean_modification_mae']:.6g} | {r['tail20_clean_modification_mae']:.6g} |")
    lines+=['','## 相近去噪量的比较','',
        '复用v3已冻结frontier/match：先形成有限强度采样的Pareto前沿，再只在正的共同全图MSE降低区间内插值。支配点剔除仅用于插值，不是调参；所有原始点保留。'
        '个体比较与12图均值曲线比较分开，均值曲线不能替代无法改善的个体。','',
        '| 加噪条件 | 比较 | 有正共同去噪区间 | 个体匹配中解耦损伤更低／相同／更高的点数 |',
        '|---|---|---:|---|']
    for kind in KINDS:
        for pair in ('coupled','unprotected'):
            comparison='decoupled_vs_'+pair
            group=[r for r in coverage if r['kind']==kind and r['comparison']==comparison]
            individual=[r for r in matched if r['kind']==kind and r['comparison']==comparison and r['group']=='individual' and r['matched_contaminated_mse_reduction_fraction']>1e-12]
            lower=sum(r['clean_mse_difference'] < -1e-15 for r in individual)
            equal=sum(abs(r['clean_mse_difference']) <=1e-15 for r in individual)
            higher=sum(r['clean_mse_difference'] >1e-15 for r in individual)
            lines.append(f'| {kind} | {comparison} | {sum(r["matched_points"]>0 for r in group)}/12 | {lower}／{equal}／{higher} |')
    lines+=['','点数来自同一照片曲线上的插值，不能当作独立样本量。48个个体比较全部保留在matched_coverage.csv；没有正共同区间的照片不进入保护效果结论，且绝不删除。',
        '自然纹理的恢复能力与保真必须同时阅读：低clean修改可能来自完全不处理，也可能伴随加噪版本没有改善；只有共同正去噪区间内才有对应损伤比较。','',
        '本次固定alpha下，解耦版在global和chirp上的12图平均恢复MSE仍高于含噪输入，说明平均意义上仍造成净损伤。'
        '在可匹配的正去噪区间内，解耦与耦合曲线大多重合，不能据此确认解耦相对耦合带来了额外的选择性保护；'
        '相对关闭保护的共同区间损伤则较低。所有这些结果仍属于后验探索，不改变主实验结论或运行参数。','',
        '## 文件与核验','',
        '- `paired_curves.csv`：576个加噪恢复／原照片损伤配对点。',
        '- `clean_controls.csv`：288个原照片运行点。',
        '- `matched_coverage.csv`：24个照片条件×2种比较，共48行，包含无法匹配的情况。',
        '- `matched_removal.csv`：个体与均值曲线的共同区间插值；没有区间时为空表。',
        '- `paired_tradeoff.png`、`individual_coverage.png`：总体曲线与所有个体覆盖。',
        '- `verification.json`：本补充自己的核验，不修改既有主实验verification.json。','',
        f'本补充核验{sum(checks.values())}/{len(checks)}项通过。数据与源码哈希详见`provenance.json`。','']
    (FOLDER/'EXPLORATORY.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    frozen,cfg=load_frozen();config_hash=sha(CONFIG);snapshot_hash=frozen['snapshot']['snapshot_sha256']
    FOLDER.mkdir(parents=True,exist_ok=True)
    dataset=records('texture_test');ids=[r['id'] for r in dataset]
    checks={};controls=[];paired=[];matched_rows=[];coverage=[];recipes=[];input_hashes=[]
    clean_lookup={};prepare_count=0
    for index,record in enumerate(dataset):
        photo=load_clean(record,DATA)
        p=prepare(photo,cfg);prepare_count+=1
        input_hashes.append(dict(image_id=record['id'],kind='clean',input_array_sha256=array_sha(photo)))
        for mode in MODES:
            for alpha in STRENGTHS:
                result=output(p,mode,alpha);error=result-photo
                r=dict(image_id=record['id'],mode=mode,strength=alpha,clean_mse=float(np.mean(error**2)),
                       clean_mae=float(np.abs(error).mean()))
                clean_lookup[(record['id'],mode,alpha)]=r;controls.append(r)
        for kind in KINDS:
            noisy,metadata,_=make_sample(photo,record['id'],kind,.12)
            recipes.append(dict(image_id=record['id'],**metadata))
            input_hashes.append(dict(image_id=record['id'],kind=kind,input_array_sha256=array_sha(noisy)))
            p=prepare(noisy,cfg);prepare_count+=1
            baseline=float(np.mean((noisy-photo)**2));case=[]
            for mode in MODES:
                for alpha in STRENGTHS:
                    result=output(p,mode,alpha);control=clean_lookup[(record['id'],mode,alpha)]
                    mse=float(np.mean((result-photo)**2))
                    row=dict(image_id=record['id'],kind=kind,mode=mode,strength=alpha,
                        contaminated_mse=mse,clean_mse=control['clean_mse'],clean_mae=control['clean_mae'],
                        input_contaminated_mse=baseline,whole_image_mse_reduction_fraction=1-mse/baseline,
                        noisy_modification_mae=float(np.abs(result-noisy).mean()),seed=metadata['seed'])
                    paired.append(row);case.append(row)
            for pair in ('coupled','unprotected'):
                points=match(case,'individual',record['id'],pair)
                for point in points:point['kind']=kind
                matched_rows.extend(points)
                modes_to_check=('decoupled',pair)
                best={mode:max(r['whole_image_mse_reduction_fraction'] for r in case if r['mode']==mode) for mode in modes_to_check}
                coverage.append(dict(image_id=record['id'],kind=kind,comparison='decoupled_vs_'+pair,
                    matched_points=len(points),decoupled_best_whole_image_mse_reduction=best['decoupled'],
                    comparator_best_whole_image_mse_reduction=best[pair],
                    reason='positive common removal interval' if points else 'no positive common whole-image removal interval'))
        print(f'exploratory DTD paired trade-off {index+1}/12 {record["id"]}',flush=True)
    means=[];groups=defaultdict(list)
    for row in paired:groups[(row['kind'],row['mode'],row['strength'])].append(row)
    for (kind,mode,alpha),group in groups.items():
        means.append(dict(kind=kind,mode=mode,strength=alpha,**{key:float(np.mean([r[key] for r in group]))
            for key in ('contaminated_mse','clean_mse','input_contaminated_mse')}))
    for kind in KINDS:
        for pair in ('coupled','unprotected'):
            points=match([r for r in means if r['kind']==kind],'subset_mean','all12',pair)
            for point in points:point['kind']=kind
            matched_rows.extend(points)
    operating=[]
    for kind in KINDS:
        for mode in MODES:
            group=[r for r in paired if r['kind']==kind and r['mode']==mode and r['strength']==cfg.strength]
            require=bool(group)
            if not require:raise RuntimeError('The frozen alpha is not on the declared diagnostic grid; refusing to choose a replacement')
            maes=[r['clean_mae'] for r in group]
            operating.append(dict(kind=kind,mode=mode,strength=cfg.strength,n_photos=len(group),
                mean_noisy_restoration_mse=float(np.mean([r['contaminated_mse'] for r in group])),
                mean_input_noisy_mse=float(np.mean([r['input_contaminated_mse'] for r in group])),
                mean_clean_modification_mse=float(np.mean([r['clean_mse'] for r in group])),
                mean_clean_modification_mae=float(np.mean(maes)),tail20_clean_modification_mae=tail_mean(maes),
                max_clean_modification_mae=max(maes)))
    write_csv(FOLDER/'clean_controls.csv',controls);write_csv(FOLDER/'paired_curves.csv',paired)
    write_csv(FOLDER/'paired_curves_mean.csv',means);write_csv(FOLDER/'matched_coverage.csv',coverage)
    fields=['group','image_id','comparison','matched_contaminated_mse_reduction_fraction','decoupled_clean_mse',
        'coupled_clean_mse','unprotected_clean_mse','clean_mse_difference','common_removal_min','common_removal_max','kind']
    write_csv(FOLDER/'matched_removal.csv',matched_rows,fields);write_csv(FOLDER/'frozen_operating_point.csv',operating)
    dump(FOLDER/'synthesis.json',recipes)
    sources=[Path(__file__).name,'v3_evidence.py','v3_experiments.py','denoiser_v3.py','local_estimation_v3.py',
        'texture_guard_v3.py','validation_data_v3.py','evaluation_v3.py','image_io_v3.py']
    provenance=dict(exploratory=True,posthoc=True,used_for_parameter_selection=False,new_operating_point_selected=False,
        original_frozen_alpha=cfg.strength,config_sha256=config_hash,frozen_snapshot_sha256=snapshot_hash,
        source_hashes={name:sha(ROOT/name) for name in sources},ids=ids,conditions=['clean','global.12','chirp.12'],
        input_preparations=prepare_count,structure_strength_grid=STRENGTHS,
        axis_definition=dict(contaminated_mse='whole-image noisy restoration MSE against original photo; not a local region loss',
            clean_mse='modification MSE on the paired original photo, same mode and alpha',
            input_contaminated_mse='whole-image added-noise input MSE against original photo'),
        reference_boundary='Original natural photos contain no newly added noise, but are not guaranteed physically noise-free.',
        matching='Reuse frozen v3_evidence.frontier/match; finite-grid Pareto interpolation only in positive common removal intervals.',
        input_array_hashes=input_hashes)
    dump(FOLDER/'provenance.json',provenance)
    checks['complete_clean_controls']=len(controls)==len({(r['image_id'],r['mode'],r['strength']) for r in controls})==288
    checks['complete_paired_curves']=len(paired)==len({(r['image_id'],r['kind'],r['mode'],r['strength']) for r in paired})==576
    checks['all_cases_retained_in_coverage']=len(coverage)==len({(r['image_id'],r['kind'],r['comparison']) for r in coverage})==48
    checks['every_input_estimated_once']=prepare_count==36
    checks['finite_nonnegative_losses']=all(np.isfinite(r[k]) and r[k]>=0 for r in paired for k in ('contaminated_mse','clean_mse','input_contaminated_mse'))
    checks['same_alpha_clean_pairing']=all(r['clean_mse']==clean_lookup[(r['image_id'],r['mode'],r['strength'])]['clean_mse'] for r in paired)
    checks['zero_strength_identity']=all(abs(r['contaminated_mse']-r['input_contaminated_mse'])<1e-12 and r['clean_mse']==0 for r in paired if r['strength']==0)
    checks['positive_common_intervals_only']=all(0<=r['common_removal_min']<r['common_removal_max'] and r['common_removal_min']-1e-12<=r['matched_contaminated_mse_reduction_fraction']<=r['common_removal_max']+1e-12 for r in matched_rows)
    checks['every_unmatched_case_explained']=all(r['matched_points']>0 or bool(r['reason']) for r in coverage)
    checks['fixed_frozen_operating_point']=all(r['strength']==cfg.strength and r['n_photos']==12 for r in operating)
    # Independently match the already-saved confirmatory texture outputs at the
    # unchanged alpha. This check does not refit or select anything.
    primary=read_csv(OUT/'textures/metrics.csv');lookup={(r['image_id'],r['kind'],r['method']):float(r['mse']) for r in primary}
    map_method=dict(coupled='v3_coupled_P0',decoupled='v3_decoupled',unprotected='v3_unprotected')
    consistent=True
    for r in paired:
        if r['strength']==cfg.strength:
            consistent &= bool(np.isclose(r['contaminated_mse'],lookup[(r['image_id'],r['kind'],map_method[r['mode']])],rtol=1e-10,atol=1e-12))
    for r in controls:
        if r['strength']==cfg.strength:
            consistent &= bool(np.isclose(r['clean_mse'],lookup[(r['image_id'],'clean',map_method[r['mode']])],rtol=1e-10,atol=1e-12))
    checks['fixed_point_matches_primary_texture_results']=consistent
    again,_=load_frozen()
    checks['frozen_configuration_and_sources_unchanged']=sha(CONFIG)==config_hash and again['snapshot']['snapshot_sha256']==snapshot_hash
    plots(means,matched_rows,coverage,ids)
    report=dict(all_passed=all(checks.values()),checks=checks,exploratory=True,
        no_parameter_or_operating_point_selection=True,counts=dict(clean_controls=288,paired_curves=576,coverage=48,
            matched_points=len(matched_rows),prepared_inputs=36),
        config_sha256=config_hash,script_sha256=sha(__file__),
        matched_cases_by_comparison=dict(Counter((r['kind']+'/'+r['comparison']) for r in coverage if r['matched_points']>0)))
    dump(FOLDER/'verification.json',report);note(operating,coverage,matched_rows,config_hash,cfg.strength,checks)
    print(json.dumps(report,indent=2),flush=True)
    if not report['all_passed']:raise RuntimeError('Exploratory diagnostic artifact verification failed')


if __name__=='__main__':main()
