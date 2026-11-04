#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:30:00
#SBATCH --output=slurm_output_%A.out

# Usage, from the root of the repository, after jobs/install_env.sh has finished and
# `wandb login` has been run once on the login node (with uvadlc_seminars activated):
#   sbatch jobs/train.sh                                   # configs/where/snellius_mig.yaml
#   sbatch jobs/train.sh optim.lr=0.05 train.max_epochs=5  # any key=value override
#   sbatch jobs/train.sh --overlay my_experiment.yaml      # an experiment on top
#
# One MIG slice is the smallest GPU ask: 9 cores + 1 MIG + 60 GiB, 64 SBU per hour. Asking
# for more cores or memory bills a bigger share of the node. The first job downloads
# CIFAR-10 into /scratch-shared/$USER/data.

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate uvadlc_seminars

set -euo pipefail

cd "$HOME/uva_dl1_seminars_2026"
# The commit being run: a checkout you forgot to pull is a classic waste of a job.
echo "commit $(git rev-parse --short HEAD) | job $SLURM_JOB_ID | host $(hostname)"

python train.py --where snellius_mig "$@"
