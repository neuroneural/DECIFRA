#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=15g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:A100:1           
#SBATCH -t 1600                     
#SBATCH -J pretrain_DECIFRA_rand                   
#SBATCH -D .                        
#SBATCH -e ./_error/error_DECIFRA_rand%A_%a.err  
#SBATCH -o ./_out/out_DECIFRA_rand%A_%a.out    
#SBATCH -A psy53c17     
#SBATCH --array=0-9

# 1 config * 10 runs each = 10 tasks. Indices 0 to 9
# This assumes SLURM array is executed with: sbatch scripts/01_batch_pretraining_DECIFRA_rand.sh

sleep 10s

echo $HOSTNAME >&2

# Run the actual job
source /data/users2/ppopov1/miniconda/bin/activate pile

CONFIGS=(
    "assets/configs/DECIFRA_MS/default.yaml"
)

# Using modulo logic
config_idx=$(($SLURM_ARRAY_TASK_ID % ${#CONFIGS[@]}))
# Integer division groups into subsequent trials 
idx=$(($SLURM_ARRAY_TASK_ID / ${#CONFIGS[@]}))  

hp_config=${CONFIGS[$config_idx]}
dataset="ukb"

# Strip the path to find a nice human-readable descriptor to name the folders (e.g., default)
config_basename=$(basename $hp_config .yaml)

echo "Running Pretrain DECIFRA_rand Config: $config_basename at Run Index: $idx"

PYTHONPATH=. python scripts/01_batch_pretraining.py \
    --idx $idx --model DECIFRA_rand --dataset $dataset --hp_config $hp_config \
    --postfix $config_basename

sleep 30s
