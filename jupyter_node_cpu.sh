#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=15g
#SBATCH -p qTRDHM
#SBATCH -t 1600                       # time in minutes
#SBATCH -J jupcpu                   # job name in SLURM
#SBATCH -D .                        # adding this means that node starting path is the path from which you run this script
#SBATCH --output=./_out/jnode-%j.out     # output file name
#SBATCH -A psy53c17                 # elpis project name, can be different for you, check you allocations at https://elpis.rs.gsu.edu/

PORT="4466"
TOKEN="VGH9W56uz3PUlilaTmobEvQoxImbYsL6DppFyCzwuCI"

# assuming you created the python env following the README; change ml4fmri to your env name if you used a different one
/data/users2/ppopov1/miniconda/bin/conda run -n pile --no-capture-output jupyter-lab \
  --ip=0.0.0.0 --no-browser \
  --ServerApp.port="${PORT:-8888}" \
  --IdentityProvider.token="${TOKEN:?Set TOKEN}" \
  --ServerApp.password='' \
  --ServerApp.allow_remote_access=True &

# print a single concise line you can copy
echo "URL: http://$(hostname -f):$PORT/lab?token=$TOKEN"

wait