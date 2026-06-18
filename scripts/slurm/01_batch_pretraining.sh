#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=15g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:A100:1           # ask for 1 A100 gpu, needed if you want to try heavier models
#SBATCH -t 1600                       # time in minutes
#SBATCH -J pretrain                   # job name in SLURM
#SBATCH -D .                        # adding this means that node starting path is the path from which you run this script
#SBATCH -e ./_error/error%A_%a.err  # errors will be written to this file. If saving this file in a separate folder, make sure the folder exists, or the job will fail
#SBATCH -o ./_out/out%A_%a.out    # output will be written to this file. If saving this file in a separate folder, make sure the folder exists, or the job will fail
#SBATCH -A psy53c17     # user group. See “requesting an account” page for list of groups


# it is a good practice to add small delay at the beginning and end of the job- helps to preserve stability of SLURM controller when large number of jobs fail simultaneously
sleep 10s

# for debugging purpose- in case the job fails, you know where to look for possible cause
echo $HOSTNAME >&2

# run the actual job
source /data/users2/ppopov1/miniconda/bin/activate pile
# Vanilla DECIFRA variants share one config; select the class via model.variant
# VARIANTS=(default noGate noGate_IMix_Res IMix IMix_Res Gated_IMix_Res)
# variant_idx=$(($SLURM_ARRAY_TASK_ID % ${#VARIANTS[@]}))
# idx=$(($SLURM_ARRAY_TASK_ID / ${#VARIANTS[@]}))
# variant=${VARIANTS[$variant_idx]}  # then: model=DECIFRA model.variant=$variant
model='DECIFRA'
dataset="ukb"
idx=$SLURM_ARRAY_TASK_ID

PYTHONPATH=. python scripts/01_pretraining.py idx=$idx model=$model dataset=$dataset

sleep 30s
