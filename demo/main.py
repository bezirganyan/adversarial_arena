import streamlit as st
import numpy as np
import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt


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
    sns.histplot(filtered.loc[:, ['score', 'score_pert']], kde=True, stat='probability', common_norm=True, ax=ax).set(title=f'Sensitivity: {sensitivity}')
    return fig

if __name__ == '__main__':
    st.title('LPIPS Scores')
    attack = st.selectbox('Attack Type', ['pgd', 'wass'])
    target_net = st.selectbox('Target Network', ['vgg', 'alex', 'squeeze'])
    distance_net = st.selectbox('Distance Network', ['vgg', 'alex', 'squeeze']  if target_net == 'vgg' else ['alex', 'squeeze'])
    results_path = f'{attack}_scores_{target_net}_{distance_net}'
    results = load_results(results_path)
    eps_range = st.slider('Max eps', 0., 1., (0.1, 0.2))
    eps = results.eps.unique()
    eps_in_range = eps[np.bitwise_and(eps_range[0] <= eps, eps < eps_range[1])]
    st.write(f'Epsilons in range: {np.sort(eps_in_range)}')
    plot = plot_lpips_distributions(results, eps_range[0], eps_range[1])
    st.pyplot(plot)
