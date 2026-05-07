#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=30g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:A100:1           
#SBATCH -t 1600                     
#SBATCH -J saliency_DECIFRA_MS                   
#SBATCH -D .                        
#SBATCH -e ./_error/error_saliency_%A_%a.err  
#SBATCH -o ./_out/out_saliency_%A_%a.out    
#SBATCH -A psy53c17     
#SBATCH --array=0-7

sleep 10s
echo $HOSTNAME >&2

source /data/users2/ppopov1/miniconda/bin/activate pile

# All channels (0 to 52, plus -1)
channels=($(seq 0 52) -1)

# Split 54 channels into 8 chunks
# Chunk size: ceil(54 / 8) = 7
chunk_size=7
start_idx=$(($SLURM_ARRAY_TASK_ID * chunk_size))

# Slice array. In bash: ${array[@]:start:length}
target_channels="${channels[@]:$start_idx:$chunk_size}"

echo "Running saliency generation for channels: $target_channels"

PYTHONPATH=. python scripts/generate_saliency.py --target_channels $target_channels

sleep 30s
