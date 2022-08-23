#!/usr/bin/env bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "adversarial-arena"

scripts/craft_adversarial_samples.sh
scripts/get_LPIPS_scores.sh results pgd_scores
scripts/get_LPIPS_scores.sh results_wass wass_scores
scripts/get_LPIPS_scores.sh results_wass wass_scores
scripts/get_LPIPS_scores.sh results_iwass iwass_scores
scripts/get_LPIPS_scores.sh results_iwass iwass_scores