#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=128g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=arctrddgxa001
#SBATCH -t 4600
#SBATCH -J pretrain_ablations
#SBATCH -D .
#SBATCH -e ./_error/error_pretrain_ablations_%A_%a.err
#SBATCH -o ./_out/out_pretrain_ablations_%A_%a.out
#SBATCH -A psy53c17
#SBATCH --array=0-109%8

# Pretraining ablations.
# Submit from the repo root: sbatch scripts/slurm/01_pretrain_ablations.sh

EXPERIMENTS=(
    # transition matrices and multistage schedule (DECIFRA_MS on UKB)
    "DECIFRA_MS/noMixed        ukb 10"                     # M_t = identity
    "DECIFRA_MS/default        ukb 10 model.variant=rand"  # M_t = identity + noise
    "DECIFRA_MS/no_sparsity    ukb 10"                     # no sparsity loss
    "DECIFRA_MS/vanilla_strong ukb 10"                     # one stage, mixing weight 1.0
    "DECIFRA_MS/vanilla_weak   ukb 10"                     # one stage, mixing weight 0.2
    # pretraining data
    "DECIFRA_MS/default ukb_half      10"                  # first half of each time series
    "DECIFRA_MS/default ukb_plus_1000 10"                  # 1000 extra training subjects
    # parcellation: AAL3 atlas instead of ICA
    "DECIFRA            ukb_aal_2000 20 train.batch_size=32"
    "DECIFRA_MS/default ukb_aal_2000 20 train.batch_size=32"
    # "DECIFRA            ukb_aal_4000 20 train.batch_size=32"
    # "DECIFRA_MS/default ukb_aal_4000 20 train.batch_size=32"
)
# vanilla DECIFRA architecture variants (not in the paper)
# for v in noGate noGate_IMix_Res IMix IMix_Res Gated_IMix_Res; do EXPERIMENTS+=("DECIFRA ukb 10 model.variant=$v"); done

# outside SLURM: list the array index range of each experiment
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    i=0
    for exp in "${EXPERIMENTS[@]}"; do
        read -r model dataset runs overrides <<< "$exp"
        echo "$i-$((i + runs - 1)): $model $dataset $overrides"
        i=$((i + runs))
    done
    echo "total: $i runs -> --array=0-$((i - 1))"
    exit 0
fi

# experiment and run index of this array task
task=$SLURM_ARRAY_TASK_ID
for exp in "${EXPERIMENTS[@]}"; do
    read -r model dataset runs overrides <<< "$exp"
    if [ $task -lt $runs ]; then
        break
    fi
    task=$((task - runs)); model=""
done
if [ -z "$model" ]; then
    echo "array index $SLURM_ARRAY_TASK_ID is past the last experiment"; exit 0
fi

sleep 10s
echo $HOSTNAME >&2

source /data/users2/ppopov1/miniconda/bin/activate pile

echo "Pretraining $model on $dataset | run $task $overrides"
PYTHONPATH=. python scripts/01_pretraining.py \
    model=$model \
    dataset=$dataset \
    idx=$task \
    $overrides \
    resume=true

sleep 30s
