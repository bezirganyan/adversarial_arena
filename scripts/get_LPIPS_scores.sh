#!/usr/bin/env bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "adversarial-arena"

adversarialfolder=$1
folderprefix=${2:-scores}

echo "Evaluating samples: TN = vgg, DN = vgg"
echo "======================================"

scores_dir="${folderprefix}_vgg_vgg"
mkdir $scores_dir
python run_LPIPS.py --res_path=$adversarialfolder --scores_path=$scores_dir --target_network=vgg --distance_network=vgg

echo "Evaluating samples: TN = vgg, DN = alex"
echo "======================================="

scores_dir="${folderprefix}_vgg_alex"
mkdir $scores_dir
python run_LPIPS.py --res_path=$adversarialfolder --scores_path=$scores_dir --target_network=vgg --distance_network=alex

echo "Evaluating samples: TN = alex, DN = alex"
echo "========================================"

scores_dir="${folderprefix}_alex_alex"
mkdir $scores_dir
python run_LPIPS.py --res_path=$adversarialfolder --scores_path=$scores_dir --target_network=alex --distance_network=alex

echo "Evaluating samples: TN = vgg, DN = squeeze"
echo "=========================================="

scores_dir="${folderprefix}_vgg_squeeze"
mkdir $scores_dir
python run_LPIPS.py --res_path=$adversarialfolder --scores_path=$scores_dir --target_network=vgg --distance_network=squeeze

echo "Evaluating samples: TN = squeeze, DN = squeeze"
echo "=============================================="

scores_dir="${folderprefix}_squeeze_squeeze"
mkdir $scores_dir
python run_LPIPS.py --res_path=$adversarialfolder --scores_path=$scores_dir --target_network=squeeze --distance_network=squeeze