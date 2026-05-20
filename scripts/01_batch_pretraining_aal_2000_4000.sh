#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=40g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:A100:1           
#SBATCH -t 1600                     
#SBATCH -J pretrain_aal_2000_4000                   
#SBATCH -D .                        
#SBATCH -e ./_error/error_aal_2000_4000_%A_%a.err  
#SBATCH -o ./_out/out_aal_2000_4000_%A_%a.out    
#SBATCH -A psy53c17     
#SBATCH --array=0-79

sleep 10s

echo $HOSTNAME >&2

# Run the actual job
source /data/users2/ppopov1/miniconda/bin/activate pile

# Task assignment logic
# 0 to 39: ukb_aal_2000
# 40 to 79: ukb_aal_4000
if [ $SLURM_ARRAY_TASK_ID -lt 40 ]; then
    dataset="ukb_aal_2000"
    local_task_id=$SLURM_ARRAY_TASK_ID
else
    dataset="ukb_aal_4000"
    local_task_id=$(($SLURM_ARRAY_TASK_ID - 40))
fi

# 2 variants * 20 models = 40 tasks per dataset.
# Modulo determines the variant:
# 0 -> vanilla (DECIFRA)
# 1 -> MS (DECIFRA_MS)
variant_idx=$(($local_task_id % 2))
idx=$(($local_task_id / 2))

if [ $variant_idx -eq 0 ]; then
    model="DECIFRA"
    hp_config_arg=""
    postfix="vanilla"
else
    model="DECIFRA_MS"
    hp_config_arg="--hp_config assets/configs/DECIFRA_MS/default.yaml"
    postfix="default"
fi

# Run pretraining
echo "Running Pretrain for $model on $dataset at Run Index: $idx (postfix: $postfix)"
PYTHONPATH=. python scripts/01_batch_pretraining.py \
    --idx $idx --model $model --dataset $dataset $hp_config_arg \
    --postfix $postfix

sleep 30s
