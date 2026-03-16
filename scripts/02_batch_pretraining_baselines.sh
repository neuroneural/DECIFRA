#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=15g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:A100:1
#SBATCH -t 1600                     
#SBATCH -J pretrain_baselines                   
#SBATCH -D .                        
#SBATCH -e ./_error/error_baselines%A_%a.err  
#SBATCH -o ./_out/out_baselines%A_%a.out    
#SBATCH -A psy53c17     
#SBATCH --array=0-49

# 5 configs (3 VAR + 2 RNN) * 10 runs each = 50 tasks. Indices 0 to 49

sleep 10s

echo $HOSTNAME >&2

# Run the actual job
source /data/users2/ppopov1/miniconda/bin/activate pile

CONFIGS=(
    "VAR:assets/configs/VAR/lag_1_full.yaml"
    "VAR:assets/configs/VAR/lag_1_multi.yaml"
    "VAR:assets/configs/VAR/lag_1_single.yaml"
    "GRU_forecaster:assets/configs/GRU_forecaster/default.yaml"
    "LSTM_forecaster:assets/configs/LSTM_forecaster/default.yaml"
)

# Using modulo logic
config_pair_idx=$(($SLURM_ARRAY_TASK_ID % ${#CONFIGS[@]}))
# Integer division groups into subsequent trials 
idx=$(($SLURM_ARRAY_TASK_ID / ${#CONFIGS[@]}))  

pair=${CONFIGS[$config_pair_idx]}
model_name=$(echo $pair | cut -d: -f1)
hp_config=$(echo $pair | cut -d: -f2)

dataset="ukb"

# Strip the path to find a nice human-readable descriptor to name the folders
config_basename=$(basename $hp_config .yaml)

echo "Running Pretrain $model_name Config: $config_basename at Run Index: $idx"

PYTHONPATH=. python scripts/01_batch_pretraining.py --idx $idx --model $model_name --dataset $dataset --hp_config $hp_config --postfix $config_basename

sleep 30s
