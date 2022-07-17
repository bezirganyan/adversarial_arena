#!/usr/bin/env bash

echo "Generating adversarial samples using PGD"
mkdir "results"
python generate.py

echo "Generating adversarial samples using Wasserstein Frank-Wolfe + Dual LMO"
mkdir "results_wass"
(cd ../fast_wasserstein_adversarial && python generate.py)
mv ../fast_wasserstein_adversarial/results_wass/ .

echo "Generating adversarial samples using Improved Wasserstein"
mkdir "results_iwass"
(cd ../improved_wasserstein && python generate.py)
mv ../improved_wasserstein/results_iwass/ .
