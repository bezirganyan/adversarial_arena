#!/usr/bin/env bash

set -o errexit

(cd .. && git clone https://github.com/bezirganyan/fast-wasserstein-adversarial.git fast-wasserstein-adversarial)
(cd ../fast-wasserstein-adversarial && git switch LPIPS_experiments)
(cd .. && git clone https://github.com/bezirganyan/improved_wasserstein.git improved_wasserstein)

mkdir -p data
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz data/imagenette2-320.tgz
(cd data && tar -xzf imagenette2-320.tgz)
rm data/imagenette2-320.tgz

conda create -n "adversarial-arena" python=3.9.7 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "adversarial-arena"
pip install -r requirements.txt

(cd ../fast-wasserstein-adversarial/sparse_tensor && python setup.py install)
