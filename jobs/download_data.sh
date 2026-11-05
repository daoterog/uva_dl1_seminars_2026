#!/bin/bash

#SBATCH --partition=rome
#SBATCH --job-name=download
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --output=slurm_output_%A.out

# Usage, once, after jobs/install_env.sh has finished:
#   sbatch jobs/download_data.sh
#
# Downloads CIFAR-10 from the Hugging Face Hub (~130 MB, well under a minute) into the
# data.root of configs/where/snellius_mig.yaml, where the training jobs look for it. It runs
# the DataModule's own prepare_data(), so the two can never disagree on the path. A download
# needs no GPU: this runs on rome, whose smallest ask (16 cores) is 16 SBU per hour.

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate uvadlc_seminars

set -euo pipefail

cd "$HOME/uva_dl1_seminars_2026"

# The Hub keeps its own download cache besides the dataset: put it on scratch too.
export HF_HOME="/scratch-shared/$USER/huggingface"

python -c "
from src.config import build_config
from src.data import CIFAR10DataModule

cfg = build_config('snellius_mig')
CIFAR10DataModule(cfg.data).prepare_data()
print(f'CIFAR-10 is ready in {cfg.data.root}')
"
