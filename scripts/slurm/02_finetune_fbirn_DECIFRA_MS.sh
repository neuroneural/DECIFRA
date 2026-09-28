#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=128g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=arctrddgxa001
#SBATCH -t 360
#SBATCH -J ft_fbirn_DECIFRA_MS
#SBATCH -D .
#SBATCH -e ./_error/error_ft_fbirn_DECIFRA_MS_%A_%a.err
#SBATCH -o ./_out/out_ft_fbirn_DECIFRA_MS_%A_%a.out
#SBATCH -A psy53c17
#SBATCH --array=0-4

# Fine-tune DECIFRA_MS on FBIRN diagnosis with the default nested CV
# (5 outer folds x 10 inner train/val repeats). The array index selects the
# OUTER FOLD, so the 5 folds run in parallel and write into one shared run dir
# (2_finetune-fbirn-DECIFRA_MS-default/); the last task to finish aggregates
# the summary. Pretrained weights come from conf/model/DECIFRA_MS/default.yaml
# (finetune.pretrained.run); see 02_finetune_fbirn_DECIFRA.sh for other options.

sleep 10s
echo $HOSTNAME >&2

source /data/users2/ppopov1/miniconda/bin/activate pile

fold=$SLURM_ARRAY_TASK_ID
echo "Fine-tuning DECIFRA_MS on FBIRN | fold=$fold"

PYTHONPATH=. python scripts/02_finetuning.py \
    model=DECIFRA_MS/default \
    dataset=fbirn \
    fold=$fold \
    resume=true

sleep 30s
