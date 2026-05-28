---
description: Guidelines for generating SLURM batch scripts
---

When generating or editing SLURM batch scripts (`.sh` files) for this project, ALWAYS observe the following project-specific guidelines:

1. **Partition Specification**: Use the partition `#SBATCH -p qTRDGPUH` if a GPU is required for the job. Гse `qTRD` for CPU jobs.
2. **GPU Specification**: Specify the exact GPU architecture explicitly using `#SBATCH --gres=gpu:A100:1` or `#SBATCH --gres=gpu:V100:1`. Never leave this as a generic `--gres=gpu:1`.
3. **Environment Setup**: Always activate the pile conda environment before running the python jobs using `source /data/users2/ppopov1/miniconda/bin/activate pile`.
4. **General Flags**: Include the standard array outputs (`-o` and `-e`), directory context (`-D .`), and account group (`-A psy53c17`).