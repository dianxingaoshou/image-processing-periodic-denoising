"""Plot measured support ablations without combining datasets or tuning models."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


KINDS = ('edge8', 'edge16', 'edge32', 'interior16', 'patch32')
METHODS = ('baseline', 'a32', 's32', 'grid', 'a32_grid', 's32_grid', 'a32_s32', 'a32_s32_grid')
SELECTED = ('baseline', 'a32', 's32', 'grid', 'a32_s32_grid')
LABELS = {'baseline': 'Baseline', 'a32': 'A32', 's32': 'S32', 'grid': 'Grid',
          'a32_grid': 'A32 + grid', 's32_grid': 'S32 + grid', 'a32_s32': 'A32 + S32',
          'a32_s32_grid': 'A32 + S32 + grid'}
COLORS = {'baseline': '#293f5b', 'a32': '#3272ad', 's32': '#c37822', 'grid': '#339b80',
          'a32_s32_grid': '#a55287'}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def _primary(rows, dataset):
    return [row for row in rows if row['dataset'] == dataset and row['method'] != 'input'
            and float(row['strength']) == 2.]


def _mean(rows, method, kind, field):
    values = [float(row[field]) for row in rows if row['method'] == method
              and row['kind'] == kind and row.get(field, '') != '']
    if not values:
        raise ValueError(f'Missing measured values for {method}/{kind}/{field}')
    return float(np.mean(values))


def _line_panel(axis, rows, field, title):
    values = []
    for method in SELECTED:
        means = [_mean(rows, method, kind, field) for kind in KINDS]
        values.extend(means)
        axis.plot(np.arange(len(KINDS)), means, marker='o', markersize=5,
                  color=COLORS[method], label=LABELS[method], linewidth=1.7)
    axis.set_xticks(np.arange(len(KINDS)), ['Edge 8px', 'Edge 16px', 'Edge 32px', 'Interior 16px', 'Patch 32px'])
    axis.set_title(title, loc='left', fontweight='bold')
    axis.set_ylabel('Mean MSE (lower is better)')
    axis.tick_params(axis='x', labelsize=9)
    positive = [value for value in values if value > 0]
    if field == 'adjacent_clean_mse' and positive and (min(positive) < 1e-12 or max(positive)/min(positive) > 1000):
        axis.set_yscale('symlog', linthresh=1e-12)
        axis.set_ylim(bottom=0.)
        axis.set_ylabel('Mean MSE (symlog; linear below 1e-12)')
    else:
        axis.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    axis.grid(axis='y', alpha=.22)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('results/support'))
    parser.add_argument('--figure', type=Path, default=Path('docs/figures/support-ablation.png'))
    args = parser.parse_args()
    check = _primary(read_csv(args.out/'check/metrics.csv'), 'BSDS')
    texture = [row for row in _primary(read_csv(args.out/'regression/metrics.csv'), 'DTD_regression')
               if row['kind'] == 'clean']
    paired = [row for row in read_csv(args.out/'check/paired_deltas.csv')
              if row['dataset'] == 'BSDS' and row['kind'] in KINDS]
    image_count = len({row['image_id'] for row in check})
    texture_count = len({row['image_id'] for row in texture})
    for method in METHODS:
        for kind in KINDS:
            values = [row for row in check if row['method'] == method and row['kind'] == kind]
            if len(values) != image_count or len({row['image_id'] for row in values}) != image_count:
                raise ValueError(f'Incomplete or duplicated independent-check rows: {method}/{kind}')
        values = [row for row in texture if row['method'] == method]
        if len(values) != texture_count or len({row['image_id'] for row in values}) != texture_count:
            raise ValueError(f'Incomplete or duplicated DTD rows: {method}')
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    _line_panel(axes[0, 0], check, 'contaminated_mse', '(a) Independent check: contaminated region')
    _line_panel(axes[0, 1], check, 'adjacent_clean_mse', '(b) Independent check: adjacent clean region')
    axes[0, 0].legend(fontsize=9, ncols=2, framealpha=.9)
    variants = [method for method in METHODS if method != 'baseline']
    counts = []; denominators = []
    for method in variants:
        measured = [row for row in paired if row['method'] == method
                    and row.get('contaminated_mse_delta', '') != ''
                    and row.get('clean_mse_delta', '') != ''
                    and row.get('adjacent_clean_mse_delta', '') != '']
        denominators.append(len(measured))
        counts.append(sum(row['joint_success'].lower() == 'true' for row in measured))
    axis = axes[1, 0]
    axis.barh(np.arange(len(variants)), counts, color='#587a9b', height=.6)
    for index, (count, denominator) in enumerate(zip(counts, denominators)):
        axis.text(count+.25, index, f'{count}/{denominator}', va='center', fontsize=9)
    axis.set_yticks(np.arange(len(variants)), [LABELS[m] for m in variants])
    axis.tick_params(axis='y', labelsize=9)
    axis.invert_yaxis()
    axis.set_xlim(0, max(denominators, default=1)+2.5)
    axis.set_xlabel('Image-condition pairs meeting all three tests')
    axis.set_title('(c) Improvement versus baseline, with clean-region constraints', loc='left', fontweight='bold', fontsize=10)
    axis.grid(axis='x', alpha=.22); axis.set_axisbelow(True)
    axis = axes[1, 1]
    means = []; tails = []
    for method in METHODS:
        values = sorted(float(row['mae']) for row in texture if row['method'] == method)
        if not values:
            raise ValueError(f'Missing DTD clean control: {method}')
        means.append(float(np.mean(values)))
        tails.append(float(np.mean(values[-max(1, int(np.ceil(.2*len(values)))):])))
    positions = np.arange(len(METHODS))
    axis.barh(positions-.16, means, height=.30, color='#587a9b', label='Mean MAE')
    axis.barh(positions+.16, tails, height=.30, color='#bf8953', label='Worst 20% mean MAE')
    axis.set_yticks(positions, [LABELS[m] for m in METHODS]); axis.invert_yaxis()
    axis.set_xlabel('Modification MAE: blue = mean; orange = worst 20% mean', fontsize=9)
    axis.set_title(f'(d) DTD regression: {texture_count} photos, no added noise', loc='left', fontweight='bold')
    axis.tick_params(axis='y', labelsize=9)
    axis.grid(axis='x', alpha=.22); axis.set_axisbelow(True)
    for axis in axes.ravel():
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'Support ablation: {image_count} independent BSDS images; amplitude = 0.12; strength = 2\n'
                 'A32: add 32px analysis  |  S32: 32px synthesis / 16px stride  |  Grid: 3x3 nonnegative envelope', fontsize=13)
    fig.supxlabel('Joint criterion compares with baseline (tolerance 1e-12); it does not establish improvement over the noisy input. Conditions are not independent photos.', fontsize=9)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=160)
    plt.close(fig)
    print(args.figure.resolve())


if __name__ == '__main__':
    main()
