#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=20g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:A100:1
#SBATCH -t 1200
#SBATCH -J ft_fbirn_DECIFRA
#SBATCH -D .
#SBATCH -e ./_error/error_ft_fbirn_DECIFRA_%A_%a.err
#SBATCH -o ./_out/out_ft_fbirn_DECIFRA_%A_%a.out
#SBATCH -A psy53c17
#SBATCH --array=0

# Fine-tune vanilla DECIFRA on FBIRN diagnosis with the default nested CV
# (5 outer folds x 10 inner train/val repeats = 50 runs).
#
# The pretrained weights to initialise from are set in the model config
# (conf/model/DECIFRA.yaml -> finetune.pretrained.run). Edit that to point at
# your chosen checkpoint, or override here, e.g.:
#   model.finetune.pretrained.run=assets/logs/1_pretrain-ukb-DECIFRA/03
#   model.finetune.pretrained.load=false      # from-scratch baseline
#
# The array index is just a fine-tune replicate id (-> log subdir <idx>); CV
# splits are fixed (seed 42) but clf-head init / batch shuffling vary, so
# multiple indices give a variance estimate. Use --array=0 for a single run,
# or e.g. --array=0-4 for 5 replicates.

sleep 10s
echo $HOSTNAME >&2

source /data/users2/ppopov1/miniconda/bin/activate pile

idx=$SLURM_ARRAY_TASK_ID
echo "Fine-tuning DECIFRA on FBIRN | replicate idx=$idx"

PYTHONPATH=. python scripts/02_finetuning.py \
    model=DECIFRA \
    dataset=fbirn \
    idx=$idx \
    resume=true

sleep 30s
