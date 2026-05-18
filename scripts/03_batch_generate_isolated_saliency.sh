#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=30g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:A100:1           
#SBATCH -t 1600                     
#SBATCH -J saliency_isolated_DECIFRA_MS                   
#SBATCH -D .                        
#SBATCH -e ./_error/error_saliency_iso_%A.err  
#SBATCH -o ./_out/out_saliency_iso_%A.out    
#SBATCH -A psy53c17

sleep 10s
echo $HOSTNAME >&2

source /data/users2/ppopov1/miniconda/bin/activate pile

target_channels="-1"

echo "Running isolated saliency generation for channels: $target_channels"

PYTHONPATH=. python scripts/generate_saliency.py --target_channels=$target_channels --experiment_type isolated

sleep 30s
