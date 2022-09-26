#!/usr/bin/env bash

set -o errexit

conda create -n "adversarial-demo" python=3.9.7 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "adversarial-demo"
pip install -r demo/requirements.txt

wget -O results_data.zip #download_link
unzip results_data.zip

