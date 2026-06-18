#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=15g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:A100:1
#SBATCH -t 1600
#SBATCH -J pretrain_sensitivity
#SBATCH -D .
#SBATCH -e ./_error/error_sensitivity_%A_%a.err
#SBATCH -o ./_out/out_sensitivity_%A_%a.out
#SBATCH -A psy53c17
#SBATCH --array=0-99

# Sensitivity sweep on the pretraining objective (addresses reviewer questions
# R2Q3 and R4Q3). For each of 5 loss-config variants, train both vanilla DECIFRA
# and DECIFRA_MS for 10 random seeds.
#
# Layout: 5 variants * 2 models * 10 seeds = 100 jobs (array 0-99).
#   variant_idx = SLURM_ARRAY_TASK_ID % 5
#   model_idx   = (SLURM_ARRAY_TASK_ID / 5) % 2
#   seed_idx    = SLURM_ARRAY_TASK_ID / 10
#
# To re-run only a subset, edit the --array range or use sbatch --array=...
# Example: sbatch --array=0,5,10,15 ... runs the (default, DECIFRA) variant
# across the first four seeds.

sleep 10s

echo $HOSTNAME >&2

source /data/users2/ppopov1/miniconda/bin/activate pile

VARIANTS=(
    "default"
    "sp_heavy"
    "sp_light"
    "tau_tight"
    "tau_loose"
)
MODELS=(
    "DECIFRA"
    "DECIFRA_MS"
)
CONFIG_DIRS=(
    "assets/configs/DECIFRA_sens"
    "assets/configs/DECIFRA_MS_sens"
)

variant_idx=$(($SLURM_ARRAY_TASK_ID % ${#VARIANTS[@]}))
model_idx=$(($SLURM_ARRAY_TASK_ID / ${#VARIANTS[@]} % ${#MODELS[@]}))
idx=$(($SLURM_ARRAY_TASK_ID / (${#VARIANTS[@]} * ${#MODELS[@]})))

variant=${VARIANTS[$variant_idx]}
model=${MODELS[$model_idx]}
config_dir=${CONFIG_DIRS[$model_idx]}
model_group=$(basename $config_dir)        # e.g. DECIFRA_sens / DECIFRA_MS_sens

dataset="ukb"

echo "Sensitivity sweep: model=$model group=$model_group variant=$variant seed_idx=$idx"

PYTHONPATH=. python scripts/01_pretraining.py \
    idx=$idx model=$model_group/$variant dataset=$dataset \
    resume=true

sleep 30s
