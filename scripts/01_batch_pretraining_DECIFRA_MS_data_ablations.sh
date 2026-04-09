#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=15g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:A100:1           
#SBATCH -t 1600                     
#SBATCH -J pretrain_DECIFRA_MS_data                 
#SBATCH -D .                        
#SBATCH -e ./_error/error_DECIFRA_data%A_%a.err  
#SBATCH -o ./_out/out_DECIFRA_data%A_%a.out    
#SBATCH -A psy53c17     
#SBATCH --array=0-39

# 2 dataset variants * 10 runs each = 20 tasks. Indices 0 to 19

sleep 10s
echo $HOSTNAME >&2

source /data/users2/ppopov1/miniconda/bin/activate pile

DATASETS=(
    "ukb_half"
    "ukb_plus_1000"
)

# Using modulo logic
ds_idx=$(($SLURM_ARRAY_TASK_ID % ${#DATASETS[@]}))
# Integer division groups into subsequent trials 
idx=$(($SLURM_ARRAY_TASK_ID / ${#DATASETS[@]}))  

dataset=${DATASETS[$ds_idx]}
hp_config="assets/configs/DECIFRA_MS/default.yaml"

echo "Running Pretrain DECIFRA_MS on Dataset: $dataset at Run Index: $idx"

PYTHONPATH=. python scripts/01_batch_pretraining.py \
    --idx $idx --model DECIFRA_MS --dataset $dataset --hp_config $hp_config \
    --postfix default

sleep 30s
