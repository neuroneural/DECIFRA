#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=25g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:A100:1
#SBATCH -t 1600                     
#SBATCH -J pretrain_ukb_exp                   
#SBATCH -D .                        
#SBATCH -e ./_error/error_exp%A_%a.err  
#SBATCH -o ./_out/out_exp%A_%a.out    
#SBATCH -A psy53c17     
#SBATCH --array=0-79

# 8 configs * 10 runs each = 80 tasks. Indices 0 to 79

sleep 10s

echo $HOSTNAME >&2

# Run the actual job
source /data/users2/ppopov1/miniconda/bin/activate pile

CONFIGS=(
    "VAR|lag_1_full"
    "VAR|lag_20_full"
    "meanGRU|all_shared"
    "meanGRU|all_independent"
    "meanLSTM|all_shared"
    "meanLSTM|all_independent"
    "GRU_forecaster|default"
    "LSTM_forecaster|default"
)

# Using modulo logic
config_idx=$(($SLURM_ARRAY_TASK_ID % ${#CONFIGS[@]}))
# Integer division groups into subsequent trials 
idx=$(($SLURM_ARRAY_TASK_ID / ${#CONFIGS[@]}))  

# Split model and config path
IFS='|' read -r model selection <<< "${CONFIGS[$config_idx]}"

dataset="ukb_exp"

# Strip the path to find a nice human-readable descriptor to name the folders
config_basename=$(basename $selection .yaml)
echo "Running Pretrain Model: $model Config: $config_basename at Run Index: $idx on $dataset"

PYTHONPATH=. python scripts/01_pretraining.py idx=$idx model=$model/$config_basename dataset=$dataset

sleep 30s
