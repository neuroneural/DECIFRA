#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=15g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:A100:1           
#SBATCH -t 1600                     
#SBATCH -J pretrain_DECIFRA_nosparsity                  
#SBATCH -D .                        
#SBATCH -e ./_error/error_nosparsity%A_%a.err  
#SBATCH -o ./_out/out_nosparsity%A_%a.out    
#SBATCH -A psy53c17     
#SBATCH --array=0-9

# 1 config * 10 runs each = 10 tasks. Indices 0 to 9

sleep 10s

echo $HOSTNAME >&2

source /data/users2/ppopov1/miniconda/bin/activate pile

CONFIGS=(
    "no_sparsity"
)

# Using modulo logic
config_idx=$(($SLURM_ARRAY_TASK_ID % ${#CONFIGS[@]}))
# Integer division groups into subsequent trials 
idx=$(($SLURM_ARRAY_TASK_ID / ${#CONFIGS[@]}))  

hp_config=${CONFIGS[$config_idx]}
dataset="ukb"

# Strip the path to find a nice human-readable descriptor to name the folders (e.g., default)
config_basename=$(basename $hp_config .yaml)

echo "Running Pretrain DECIFRA_MS Config: $config_basename at Run Index: $idx"

PYTHONPATH=. python scripts/01_pretraining.py \
    idx=$idx model=DECIFRA_MS/$config_basename dataset=$dataset

sleep 30s
