#!/usr/bin/env bash

scripts/craft_adversarial_samples.sh
scripts/get_LPIPS_scores.sh results pgd_scores
scripts/get_LPIPS_scores.sh results_wass wass_scores
scripts/get_LPIPS_scores.sh results_iwass iwass_scores