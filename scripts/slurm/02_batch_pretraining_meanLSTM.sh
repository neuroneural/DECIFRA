#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=15g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:A100:1           
#SBATCH -t 1600                     
#SBATCH -J pretrain_meanLSTM                   
#SBATCH -D .                        
#SBATCH -e ./_error/error_meanLSTM%A_%a.err  
#SBATCH -o ./_out/out_meanLSTM%A_%a.out    
#SBATCH -A psy53c17     
#SBATCH --array=0-39

# 4 configs * 10 runs each = 40 tasks. Indices 0 to 39

sleep 10s

echo $HOSTNAME >&2

# Run the actual job
source /data/users2/ppopov1/miniconda/bin/activate pile

CONFIGS=(
    "all_shared"
    "all_independent"
    "shared_features_independent_dynamics"
    "independent_features_shared_dynamics"
)

# Using modulo logic
config_idx=$(($SLURM_ARRAY_TASK_ID % ${#CONFIGS[@]}))
# Integer division groups into subsequent trials 
idx=$(($SLURM_ARRAY_TASK_ID / ${#CONFIGS[@]}))  

hp_config=${CONFIGS[$config_idx]}
dataset="ukb"

# Strip the path to find a nice human-readable descriptor to name the folders
config_basename=$(basename $hp_config .yaml)

echo "Running Pretrain meanLSTM Config: $config_basename at Run Index: $idx"

PYTHONPATH=. python scripts/01_pretraining.py idx=$idx model=meanLSTM/$config_basename dataset=$dataset

sleep 30s
