#!/usr/bin/env bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "adversarial-arena"

echo "Generating adversarial samples using PGD"
mkdir "results"
python generate.py

echo "Generating adversarial samples using Wasserstein Frank-Wolfe + Dual LMO"
mkdir "results_wass"
(cd ../fast_wasserstein_adversarial && python generate.py)
(cd ../fast_wasserstein_adversarial && python generate.py --eps_start=0.0005 --eps_end=0.05 --eps_count=10)
mv ../fast_wasserstein_adversarial/results_wass/ .

echo "Generating adversarial samples using Improved Wasserstein"
mkdir "results_iwass"
(cd ../improved_wasserstein && python generate.py)
(cd ../improved_wasserstein && python generate.py --eps_start=0.0005 --eps_end=0.05 --eps_count=10)
mv ../improved_wasserstein/results_iwass/ .
