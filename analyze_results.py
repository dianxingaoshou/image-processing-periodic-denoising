"""Summarize verified numeric outputs and render scientific comparisons."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ORDER = ['input','global_fixed','global_adaptive','published_arpenos','local_fixed128',
         'without_scale_selection','without_conservative','local_adaptive']
LABELS = dict(input='Input',global_fixed='Global fixed',global_adaptive='Starter adaptive',
              published_arpenos='ARPENOS port',local_fixed128='Local fixed 128',
              without_scale_selection='Forced scale 128',without_conservative='No conservative',
              local_adaptive='Full method')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def numeric(rows,key):
    return np.array([float(r[key]) for r in rows if r.get(key,'')!=''],float)


def aggregate(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[(r['kind'],r['method'])].append(r)
        if r['kind']!='clean':
            groups[('all_noisy',r['method'])].append(r)
    summary=[]
    for (kind,method),group in sorted(groups.items()):
        valid = [r for r in group if not r.get('error')]
        row=dict(kind=kind,method=method,n=len(group),valid=len(valid),exceptions=len(group)-len(valid))
        for key in ('psnr_db','ssim','mse','mae','contaminated_mse','clean_mse','clean_modification_mae',
                    'runtime_ms','degraded','output_projection_fraction'):
            values=numeric(valid,key)
            row['mean_'+key]=float(np.mean(values)) if len(values) else ''
        row['median_runtime_ms']=float(np.median(numeric(valid,'runtime_ms'))) if valid else ''
        summary.append(row)
    return summary


def paired_bootstrap(rows):
    lookup={(r['image_id'],r['kind'],r['amplitude'],r['method']):r for r in rows if not r.get('error')}
    result=[]
    rng=np.random.default_rng(20261116)
    for kind in ('global','local','chirp','orientation'):
        for baseline in ORDER[1:-1]:
            differences=defaultdict(list)
            for key,row in lookup.items():
                image_id,k,amp,method=key
                if k!=kind or method!='local_adaptive': continue
                other=lookup.get((image_id,k,amp,baseline))
                if other:
                    differences[image_id].append(float(row['psnr_db'])-float(other['psnr_db']))
            d=np.array([np.mean(v) for v in differences.values()])
            if len(d):
                samples=d[rng.integers(0,len(d),size=(2000,len(d)))].mean(axis=1)
                lo,hi=np.quantile(samples,[.025,.975])
                result.append(dict(kind=kind,baseline=baseline,n_images=len(d),
                    mean_delta_psnr=float(d.mean()),image_bootstrap_low=float(lo),image_bootstrap_high=float(hi)))
    return result


def plots(root,summary):
    folder=root/'figures'; folder.mkdir(exist_ok=True)
    lookup={(r['kind'],r['method']):r for r in summary}
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    kinds=('global','local','chirp','orientation')
    methods=('global_adaptive','published_arpenos','local_fixed128','without_conservative','local_adaptive')
    x=np.arange(4)
    for j,m in enumerate(methods):
        gains=[lookup[(k,m)]['mean_psnr_db']-lookup[(k,'input')]['mean_psnr_db'] for k in kinds]
        axes[0].bar(x+(j-2)*.15,gains,width=.145,label=LABELS[m])
    axes[0].set_xticks(x,kinds);axes[0].set_ylabel('Mean PSNR improvement (dB)');axes[0].axhline(0,color='black',lw=.6)
    clean_errors=[lookup[('clean',m)]['mean_mae'] for m in methods]
    axes[1].bar(np.arange(len(methods)),clean_errors)
    axes[1].set_xticks(np.arange(len(methods)),[LABELS[m] for m in methods],rotation=24,ha='right')
    axes[1].set_ylabel('MAE introduced on clean controls')
    axes[0].legend(fontsize=8)
    fig.savefig(folder/'overview.png',dpi=180);plt.close(fig)
    for example in sorted((root/'examples').glob('*')):
        if not example.is_dir():continue
        names=['reference','input','global_adaptive','published_arpenos','local_fixed128','local_adaptive','without_conservative']
        fig,axes=plt.subplots(2,4,figsize=(13,6.8),layout='constrained')
        reference=np.load(example/'reference.npy')
        for ax,name in zip(axes.flat,names):
            path=example/f'{name}.npy'
            if not path.exists():ax.axis('off');continue
            a=np.load(path);mse=np.mean((reference-a)**2)
            psnr=-10*np.log10(mse) if mse else float('inf')
            ax.imshow(a,cmap='gray',vmin=0,vmax=1)
            ax.set_title(('Reference' if name=='reference' else LABELS[name])+f'  {psnr:.2f} dB',fontsize=9)
            ax.axis('off')
        axes.flat[-1].axis('off')
        fig.suptitle(example.name+' | shared display range [0, 1]',fontsize=11)
        fig.savefig(folder/f'{example.name}_comparison.png',dpi=150);plt.close(fig)
        path=example/'diagnostics.npz'
        if not path.exists():continue
        d=np.load(path)
        fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
        keys=['residual','suppression','selected_scale','frequency_y','frequency_x','confidence']
        limits=[(-.18,.18),(0,1),(0,256),(-.2,.2),(-.2,.2),(0,1)]
        titles=['Subtracted residual','Suppression weight','Weighted selected scale (pixels)',
                'fy (cycles/pixel)','fx (cycles/pixel)','Reliability score (not probability)']
        for ax,key,lim,title in zip(axes.flat,keys,limits,titles):
            if key not in d:ax.axis('off');continue
            im=ax.imshow(d[key],vmin=lim[0],vmax=lim[1],cmap='coolwarm' if key in ('residual','frequency_y','frequency_x') else 'viridis')
            ax.set_title(title,fontsize=9);ax.axis('off');fig.colorbar(im,ax=ax,shrink=.75)
        fig.suptitle(example.name+' | weighted overlap maps')
        fig.savefig(folder/f'{example.name}_diagnostics.png',dpi=150);plt.close(fig)


def write_note(root,summary,validation):
    lookup={(r['kind'],r['method']):r for r in summary}
    lines=['# 代码实现与结果验证','',
           f"正式测试包含{validation['noisy_samples']}个含噪样本、{validation['clean_controls']}个无噪声对照，"
           f"共{validation['actual_rows']}条方法结果。完整性检查通过，程序异常{validation['algorithm_exceptions']}条。",'',
           '开发与测试按原图分离，参数在开发集冻结后运行测试。以下为浮点图像指标，显示范围统一为[0,1]。','',
           '## 各条件平均 PSNR','',
           '| 方法 | 全局定频 | 局部定频 | 频率渐变 | 方向变化 | 无噪声对照 MAE |',
           '|---|---:|---:|---:|---:|---:|']
    for method in ORDER:
        values=[f"{lookup[(kind,method)]['mean_psnr_db']:.2f}" for kind in ('global','local','chirp','orientation')]
        mae=lookup[('clean',method)]['mean_mae']
        lines.append('| '+LABELS[method]+' | '+' | '.join(values)+f' | {mae:.6f} |')
    lines += ['', '各列只比较同一退化条件中的方法。局部条件噪声总能量更低，不能把其较高的原始PSNR解释为算法优势。', '',
              '## 质量退化和速度','',
              '| 方法 | 含噪样本输出比输入更差比例 | 平均时间 ms |',
              '|---|---:|---:|']
    for method in ORDER[1:]:
        row=lookup[('all_noisy',method)]
        lines.append(f"| {LABELS[method]} | {100*row['mean_degraded']:.1f}% | {row['mean_runtime_ms']:.1f} |")
    lines += ['', '质量退化定义为输出MSE高于输入MSE加1e-8。程序异常与质量退化分开统计。时间为本机单次运行近似值，不能作为严格速度排名。', '',
              '## 结果的适用边界','',
              '- 不能从实现了尺度选择就断言其有效。`paired_comparisons.csv` 单独给出完整方法相对固定尺度与纯尺度消融的配对差异和按原图自举区间。',
              '- 保守策略会减少去噪量。`../supplement/matched_removal.csv` 仅在两条曲线的共同去噪区间内比较干净区损伤；`matched_coverage.csv` 记录无法匹配的案例。该补充分析只有预先固定的10张测试图，不能代替全部测试。',
              '- 附件中的单正弦纹理失败已精确复现，保守策略能原样保留该输入；这不代表解决全部纹理误滤。网格、多频和渐变周期纹理仍有误删，见补充纹理表及图。',
              '- 同样的像素既可解释为真实周期纹理，也可解释为平滑背景上的干扰；保存的双解释反例使用相同输入与输出哈希。该歧义无法由本方法保证消除。',
              '- 当前只验证合成退化，ARPENOS为明确记录数值差异的作者源码指导移植，未复现原论文的数值。', '',
              '## 查看文件','',
              '- 原始指标：`metrics.csv`；按条件汇总：`summary.csv`；完整性检查：`validation.json`。',
              '- 固定前三个测试ID的对比图与诊断：`figures/`。',
              '- 按最大退化选出的失败图与冻结参数重跑核验：`failure_audit/`。',
              '- 配置、代码与划分的独立交付检查：`../verification.json`。',
              '- 一键重跑方法见项目根目录 `README.md` 和 `run_all.ps1`。','']
    (root/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,default=Path('results/main'))
    args=p.parse_args();root=args.results
    with (root/'metrics.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    cfg=json.loads((root/'run_config.json').read_text(encoding='utf-8'))
    n_images=len(cfg['test_ids']);expected=n_images*13*len(ORDER)
    counts=defaultdict(int)
    for row in rows:counts[row['method']]+=1
    keys=[(r['image_id'],r['kind'],r['amplitude'],r['method']) for r in rows]
    failures=[r for r in rows if r.get('error')]
    validation=dict(expected_rows=expected,actual_rows=len(rows),unique_rows=len(set(keys)),
                    complete=len(rows)==expected and len(set(keys))==expected,
                    method_counts=dict(counts),algorithm_exceptions=len(failures),
                    noisy_samples=n_images*12,clean_controls=n_images,
                    degraded_definition='output MSE > input MSE + 1e-8; separate clean MAE also reported',
                    timing_note='single-run wall time on this machine, shared cached preparation charged to each method; not a speed benchmark')
    (root/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    if not validation['complete'] or failures:raise RuntimeError(validation)
    summary=aggregate(rows);write_csv(root/'summary.csv',summary)
    write_csv(root/'paired_comparisons.csv',paired_bootstrap(rows))
    plots(root,summary)
    write_note(root,summary,validation)
    for row in summary:
        if row['kind'] in ('all_noisy','clean'):
            print(row['kind'],row['method'],'PSNR',round(row['mean_psnr_db'],3),'MAE',round(row['mean_mae'],6),
                  'degraded',round(row['mean_degraded'],3))


if __name__=='__main__':main()
