#!/bin/bash
# Train on a full A100 (partition gpu_a100). Only worth it when bench.py shows the model
# keeps a whole A100 busy; CIFAR-10 with ResNet-18 mostly does not, and a MIG slice is
# billed for a fraction of the card.
#
#   sbatch jobs/train_a100.sh
#
# A gpu_a100 job is billed for its largest share of the node: 1 GPU, or 18 cores, or
# 120 GiB of memory. Asking for more than one share of cores or memory bills you for a
# second GPU you are not using.
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --job-name=dl1-train
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=TIME_LIMIT_90,FAIL

source jobs/_env.sh
"$PYTHON" train.py --where snellius_a100 "$@"
