"""Plot measured focused-study rows; no inference or parameter selection."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('results/refinement'))
    parser.add_argument('--figure', type=Path, default=Path('docs/figures/refinement-risk.png'))
    args = parser.parse_args()
    check = load(args.out / 'check/metrics.csv')
    regression = load(args.out / 'regression/metrics.csv')
    kinds = ('global', 'local', 'chirp', 'orientation', 'edge_strip', 'irregular')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    colors = {'coupled_baseline': '#3268a8', 'structure_risk': '#d37b21'}
    labels = {'coupled_baseline': 'Coupled baseline', 'structure_risk': 'New risk (experimental)'}
    for method, offset in (('coupled_baseline', -.18), ('structure_risk', .18)):
        means = [np.mean([float(r['psnr_db']) for r in check if r['method'] == method
                        and float(r['strength']) == 2 and r['kind'] == kind]) for kind in kinds]
        axes[0].bar(np.arange(len(kinds)) + offset, means, .36, label=labels[method], color=colors[method])
    axes[0].set_xticks(np.arange(len(kinds)), ['Global', 'Local', 'Chirp', 'Direction', 'Edge strip', 'Irregular'], rotation=25)
    axes[0].set_ylabel('Mean PSNR (dB), higher is better')
    axes[0].set_title('Independent check: 4 BSDS images, amplitude 0.12')
    axes[0].legend(fontsize=8)
    clean = {method: {r['image_id']: float(r['mae']) for r in regression if r['method'] == method
                    and float(r['strength']) == 2 and r['kind'] == 'clean'} for method in colors}
    ids = sorted(clean['coupled_baseline'])
    x = [clean['coupled_baseline'][i] for i in ids]
    y = [clean['structure_risk'][i] for i in ids]
    limit = max(x+y) * 1.08
    axes[1].plot([0, limit], [0, limit], '--', color='#888888', linewidth=1)
    axes[1].scatter(x, y, color='#d37b21', s=35)
    for i, xx, yy in zip(ids, x, y):
        if max(xx, yy) > .08:
            axes[1].annotate(i.removeprefix('dtd_').rsplit('_', 1)[0], (xx, yy), xytext=(4, 4), textcoords='offset points', fontsize=8)
    axes[1].set(xlabel='Baseline clean-photo modification MAE', ylabel='New-risk modification MAE',
                title='DTD regression: 12 photos, no added noise', xlim=(-.003, limit), ylim=(-.003, limit))
    for axis in axes:
        axis.grid(axis='y', alpha=.2); axis.set_axisbelow(True)
    fig.suptitle('Risk-only ablation: identical candidate residuals; strength = 2', fontsize=12)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=160)
    plt.close(fig)
    print(args.figure.resolve())


if __name__ == '__main__':
    main()
