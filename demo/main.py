import io
import streamlit as st
import numpy as np
import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


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

@st.experimental_memo
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

@st.experimental_memo
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
    results_path = './png/' # Streamlit does not support -- arguments, TODO - find another way to pass this

    st.title(' Summary of Perceptual Similarity metrics for adversarial attacks')
    st.header('1. Transferability of adversarial attacks on LPIPS metric')
    st.write("""Our hypothesis is that since LPIPS works on top of convolutional networks desinged for image
    classification (i.e. VGG, ResNet, AlexNet), then adversarial samples crafted for those networks will fool LPIPS as 
    well. In theory LPIPS scores must be higher on adversarial samples than random perturnations
    (from the same distribution), as they are optimized to maximize other activations for incorrect classification.""")

    st.write("""To test out hypothesis we design the following experiment. We craft adversarial samples for some source
    images, and also craft samples that shall be visually similar to the adversarial samples but still be classified
    with the same label as the original image (fake adversaries). Then, we will compare the distributions of the scores
    and if there is a significant difference in distributions, our hypothesis will be verified. """)
    
    st.write(r"""As a set of source images we take the Imagenette dataset, which is a subsample of 10 classes from the
    Imagenet dataset. We craft the adversarial attacks using the untargeted Projected Gradient Descent method, with
    $\epsilon \in [0,1]$. To craft the fake adversaries we take a random perturbation used for crafting the adversarial samples and add to source images as:""")

    st.latex(r'f_i = C_{[0, 1]} [x_i + \eta_j]')

    st.write(r"""Where $a_i$ are the adversarial samples, $x_i$ are the source images,
    ${\eta_j}_{}$ is a random perturbation from other adversarial samples, and $C
    (\cdot)$ is a clipping function. For our experiments we craft 10 fake
    adversaries for each source image, by adding 10 different perturbations form
    adversarial samples. After getting the distributions we will need to conclude if LPIPS is sensitive to non-targeted
    adversarial attacks. For that we define the $\delta$-sensitivity. """)

    st.write(r"""_**Definition 1.** Let $\delta>0, x$ an image and $x^{\prime}$ the same image, perturbed by a change from a
       family of adversarial attacks. We say that $d$ is $\delta-$ sensitive to this (non-targeted) attack if 
       $\mathbb{E}\left[\left|d\left(x, x^{\prime}\right)-d(x, f)\right|\right]$.
       Alternatively $: \mathbb{E}\left[\frac{\left|d\left(x, x^{\prime}\right)-d(x, f)\right|}{d\left(x, x^{\prime}\right)}\right]>\delta$_""")

    st.write("""Results can be explored in the **Demo 1** section below.""")

    st.subheader('Demo 1: LPIPS distance distributions on adversarial samples and randomly perturbed images')

    st.write("""In this demo we compare the LIPIPS distance distributions of source images with both adversarial samples
    (blue) and fake adversaries (orange).""")


    attack = st.selectbox('Attack Type', ['pgd', 'wass', 'iwass'])
    target_net = st.selectbox('Target Network', ['vgg', 'alex', 'squeeze'])
    distance_net = st.selectbox('Distance Network', ['vgg', 'alex', 'squeeze']  if target_net == 'vgg' else ['alex', 'squeeze'])
    missclassification_rates = pd.read_csv(f'./missclassification_rates.csv')
    res_filtered = missclassification_rates[
        (missclassification_rates['attack'] == attack) &
        (missclassification_rates['target_net'] == target_net) &
        (missclassification_rates['distance_net'] == distance_net)]
    eps = np.sort(res_filtered.eps.unique())
    sel_eps = float(st.selectbox('Select epsilon', eps))
    res_in_eps = res_filtered[res_filtered['eps'] == sel_eps]
    st.write(f"Missclassification rate: {   res_in_eps.adv_miss_rate.mean()}")
    st.write(f"Shuffled perturbation Missclassification rate: {res_in_eps.prt_miss_rate.mean()}")
    fig, ax = plt.subplots()
    img = mpimg.imread(f'{results_path}/{attack}_{target_net}_{distance_net}_{sel_eps:.3f}.png')
    plot = ax.imshow(img, interpolation='nearest')
    ax.axis('off')
    st.pyplot(fig)


    st.header('2. Comparing Fast-Wasserstein with Improved Wasserstein')
    metric = option = st.selectbox(
     'Metric',
     ('missclassification', 'duration'))
    fig, ax = plt.subplots()
    img = mpimg.imread(f'{results_path}/comparison_{metric}.png')
    ax.imshow(img, interpolation='nearest')
    ax.axis('off')
    st.pyplot(fig)
