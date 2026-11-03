#!/bin/bash
# Train on one MIG slice of an A100 (partition gpu_mig): the default for this course.
#
#   sbatch jobs/train_mig.sh                                   # the snellius_mig profile
#   sbatch jobs/train_mig.sh --overlay configs/failures/01_cpu_bound_loader.yaml
#   sbatch jobs/train_mig.sh data.batch_size=256 train.max_epochs=10
#
# The cores and memory a slice comes with are set by the partition. Check them with
# `scontrol show partition gpu_mig` and keep --cpus-per-task and the profile's
# data.num_workers (cores - 1) in line with what it reports.
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --job-name=dl1-train
#SBATCH --output=slurm-%j.out
# A walltime kill looks like a clean finish in wandb (Lightning catches the signal and
# exits normally). The mail at 90% of the walltime is how you find out.
#SBATCH --mail-type=TIME_LIMIT_90,FAIL

source jobs/_env.sh
"$PYTHON" train.py --where snellius_mig "$@"
