#!/usr/bin/env bash

echo "Generating adversarial samples using PGD"
python generate.py

echo "Generating adversarial samples using Wasserstein Frank-Wolfe + Dual LMO"
(cd ../fast_wasserstein_adversarial && python generate.py)
mv ../../fast_wasserstein_adversarial/results_wass/ .
cp results/pgd_*_dtl.pt results_wass/