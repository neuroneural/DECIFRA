#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=128g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=arctrddgxa001
#SBATCH -t 1600
#SBATCH -J pretrain_DECIFRA_MS
#SBATCH -D .
#SBATCH -e ./_error/error_pretrain_DECIFRA_MS_%A_%a.err
#SBATCH -o ./_out/out_pretrain_DECIFRA_MS_%A_%a.out
#SBATCH -A psy53c17
#SBATCH --array=0-19

# Pretrain multistage DECIFRA_MS on UKB.
# Submit from the repo root: sbatch scripts/slurm/01_pretrain_DECIFRA_MS.sh

sleep 10s
echo $HOSTNAME >&2

source /data/users2/ppopov1/miniconda/bin/activate pile

PYTHONPATH=. python scripts/01_pretraining.py \
    model=DECIFRA_MS/default \
    dataset=ukb \
    idx=$SLURM_ARRAY_TASK_ID \
    resume=true

sleep 30s
