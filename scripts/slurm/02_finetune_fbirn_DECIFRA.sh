#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=128g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=arctrddgxa001
#SBATCH -t 360
#SBATCH -J ft_fbirn_DECIFRA
#SBATCH -D .
#SBATCH -e ./_error/error_ft_fbirn_DECIFRA_%A_%a.err
#SBATCH -o ./_out/out_ft_fbirn_DECIFRA_%A_%a.out
#SBATCH -A psy53c17
#SBATCH --array=0-4

# Fine-tune vanilla DECIFRA on FBIRN diagnosis with the default nested CV
# (5 outer folds x 10 inner train/val repeats). The array index selects the
# OUTER FOLD, so the 5 folds run in parallel and write into one shared run dir
# (2_finetune-fbirn-DECIFRA/); the last task to finish aggregates the summary.
#
# To run all folds in a single job instead, drop --array and pass fold=null
# (the config default). To run one fold: --array=2 (=> fold=2).
#
# Pretrained weights come from the model config
# (conf/model/DECIFRA.yaml -> finetune.pretrained.run) and are recorded in each
# cell's model_config.yaml. A dir holds one source; for another one add a
# postfix, e.g. model.finetune.pretrained.load=false postfix=scratch.

sleep 10s
echo $HOSTNAME >&2

source /data/users2/ppopov1/miniconda/bin/activate pile

fold=$SLURM_ARRAY_TASK_ID
echo "Fine-tuning DECIFRA on FBIRN | fold=$fold"

PYTHONPATH=. python scripts/02_finetuning.py \
    model=DECIFRA \
    dataset=fbirn \
    fold=$fold \
    resume=true

sleep 30s
