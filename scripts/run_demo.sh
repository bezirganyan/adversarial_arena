#!/usr/bin/env bash

set -o errexit

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "adversarial-demo"

streamlit run demo/main.py