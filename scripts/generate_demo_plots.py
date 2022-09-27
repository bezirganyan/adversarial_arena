#!/usr/bin/env python
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


def load_results(results_path):
    if not results_path:
        return None
    res_files = os.listdir(results_path)
    results = pd.read_csv(f'{results_path}/{res_files[0]}', index_col=0)
    for f in res_files[1:]:
        t = pd.read_csv(f'{results_path}/{f}', index_col=0)
        results = pd.concat([results, t])
    return results

def compute_sensitivity(data):
    return ((data.score - data.score_pert) / data.score).mean()

def plot_lpips_distributions(df: pd.DataFrame, min_eps: float = 0.0, max_eps : float = 1.0) -> None:
    if df is None:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    filtered = df[df['miss'] & ~ df['miss_pert'] & (df['eps'] < max_eps) & (df['eps'] > min_eps)]
    sensitivity = compute_sensitivity(filtered)
    sns.histplot(filtered.loc[:, ['score', 'score_pert']], kde=True, stat='probability', common_norm=True, ax=ax, alpha=0.6).set(title=f'Sensitivity: {sensitivity}')
    ax.set_xlabel('LPIPS score')
    ax.legend(['Perturbed Images', 'Adversarial Images'])
    return fig

def plot_wass_comparisons(wass_df: pd.DataFrame, iwass_df: pd.DataFrame, metric='missclassficiation') -> None:
    if wass_df is None or iwass_df is None:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    if metric == 'missclassification':
        (wass_df.groupby('eps')['miss'].sum() / 9216 * 100).plot.line(label='FW+LMO', ax=ax)
        (iwass_df.groupby('eps')['miss'].sum() / 9216 * 100).plot.line(label='Improved Wasserstein', ax=ax)
        plt.legend()
    elif metric == 'duration':
        (wass_df.groupby('eps')['dur'].mean()).plot.line(label='FW+LMO', ax=ax)
        (iwass_df.groupby('eps')['dur'].mean()).plot.line(label='Improved Wasserstein', ax=ax)
        plt.legend()
    else:
        raise ValueError(f'Metric {metric} not supported')
    return fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='.')
    parser.add_argument('--output_dir', type=str, default='plots_output')
    parser.add_argument('--output_formats', type=str, default='png', help='Comma separated list of output formats')
    args = parser.parse_args()

    for fm in args.output_formats.split(','):
        Path(f'{args.output_dir}/{fm}/').mkdir(parents=True, exist_ok=True)

    for attack in tqdm(['pgd', 'wass', 'iwass']):
        for target_net in ['vgg', 'alex', 'squeeze']:
            for distance_net in ['vgg', 'alex', 'squeeze']:
                if target_net != 'vgg' and distance_net != target_net:
                    continue

                results_path = f'{args.results_dir}/{attack}_scores_{target_net}_{distance_net}'
                results = load_results(results_path)
                eps = np.sort(results.eps.unique())
                for sel_eps in eps:
                    res_in_eps = results[(results['eps'] >= (sel_eps - 1e-6)) & (results['eps'] <= (sel_eps + 1e-6))]
                    adv_miss_rate = res_in_eps.miss.sum() / res_in_eps.miss.shape[0]
                    prt_miss_rate = res_in_eps.miss_pert.sum() / res_in_eps.miss_pert.shape[0]
                    plot = plot_lpips_distributions(results, sel_eps - 1e-6, sel_eps + 1e-6)
                    for fm in args.output_formats.split(','):
                        plot.savefig(f'{args.output_dir}/{fm}/{attack}_{target_net}_{distance_net}_{sel_eps:.3f}.{fm}', dpi=300, format=fm)
    wass_df = load_results(f'{args.results_dir}/comparison_wass')
    iwass_df = load_results(f'{args.results_dir}/comparison_iwass')
    for metric in ['missclassification', 'duration']:
        plot = plot_wass_comparisons(wass_df, iwass_df, metric=metric)
        for fm in args.output_formats.split(','):
            plot.savefig(f'{args.output_dir}/{fm}/comparison_{metric}.{fm}', dpi=300, format=fm)