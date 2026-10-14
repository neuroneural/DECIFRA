#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=128g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=arctrddgxa001
#SBATCH -t 120
#SBATCH -J ft_fbirn_runs
#SBATCH -D .
#SBATCH -e ./_error/error_ft_fbirn_runs_%A_%a.err
#SBATCH -o ./_out/out_ft_fbirn_runs_%A_%a.out
#SBATCH -A psy53c17
#SBATCH --array=0-149%8

# Fine-tune every pretrained  model on FBIRN diagnosis with the default nested CV

sleep 10s
echo $HOSTNAME >&2

source /data/users2/ppopov1/miniconda/bin/activate pile

# (model config, pretraining run dir) per pretrained run
SOURCES=()
for r in $(seq -w 0 19); do SOURCES+=("DECIFRA 1_pretrain-ukb-DECIFRA/$r"); done
for r in $(seq 10 19); do SOURCES+=("DECIFRA_MS/default 1_pretrain-ukb-DECIFRA_MS-default/$r"); done

N_FOLDS=5
read model run_dir <<< "${SOURCES[$((SLURM_ARRAY_TASK_ID / N_FOLDS))]}"
fold=$((SLURM_ARRAY_TASK_ID % N_FOLDS))
run=$(basename $run_dir)
echo "Fine-tuning $model from $run_dir on FBIRN | fold=$fold"

PYTHONPATH=. python scripts/02_finetuning.py \
    model=$model \
    dataset=fbirn \
    +dataset@test_sets.holdout=fbirn_holdout \
    model.finetune.pretrained.run=assets/logs/$run_dir \
    postfix=pt$run \
    fold=$fold \
    train.save_best=false \
    resume=true

sleep 30s
