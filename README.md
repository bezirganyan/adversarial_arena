# Adversarial Arena
Comparison of several adversarial attacks

Note that for wasserstein attack, the code is implemented in [this](https://github.com/bezirganyan/fast-wasserstein-adversarial) fork, a pull request is currently made for merging with the original repo. 

## Installation

Install miniconda/anaconda, and create an environment with:

```
conda create -n adv_arena39 python=3.9.7 anaconda
conda activate adv_arena39
pip install -r adversarial_arena/requirements.txt
```

## Data

We use ImageNette for crafting adversarial samples. You can get the dataset with 

```
mkdir data
cd data
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xf imagenette2-320.tgz
```