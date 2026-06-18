#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=15g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:A100:1           
#SBATCH -t 1600                     
#SBATCH -J pretrain_VAR                   
#SBATCH -D .                        
#SBATCH -e ./_error/error_VAR%A_%a.err  
#SBATCH -o ./_out/out_VAR%A_%a.out    
#SBATCH -A psy53c17     
#SBATCH --array=0-59

# 6 configs * 10 runs each = 60 tasks. Indices 0 to 59

sleep 10s

echo $HOSTNAME >&2

# Run the actual job
source /data/users2/ppopov1/miniconda/bin/activate pile

CONFIGS=(
    "assets/configs/VAR/lag_5_full.yaml"
    "assets/configs/VAR/lag_5_single.yaml"
    "assets/configs/VAR/lag_5_multi.yaml"
    "assets/configs/VAR/lag_20_full.yaml"
    "assets/configs/VAR/lag_20_single.yaml"
    "assets/configs/VAR/lag_20_multi.yaml"
)

# Using modulo logic
config_idx=$(($SLURM_ARRAY_TASK_ID % ${#CONFIGS[@]}))
# Integer division groups into subsequent trials 
idx=$(($SLURM_ARRAY_TASK_ID / ${#CONFIGS[@]}))  

hp_config=${CONFIGS[$config_idx]}
dataset="ukb"

# Strip the path to find a nice human-readable descriptor to name the folders
config_basename=$(basename $hp_config .yaml)

echo "Running Pretrain VAR Config: $config_basename at Run Index: $idx"

PYTHONPATH=. python scripts/01_pretraining.py idx=$idx model=VAR/$config_basename dataset=$dataset

sleep 30s
