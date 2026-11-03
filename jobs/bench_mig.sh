#!/bin/bash
# Run bench.py on a MIG slice: the three rows, then whatever you pass.
#
#   sbatch jobs/bench_mig.sh                                              # real/replay/loader
#   sbatch jobs/bench_mig.sh --overlay configs/failures/01_cpu_bound_loader.yaml
#   sbatch jobs/bench_mig.sh --rows real --ladder 64,128,256,512,1024
#   sbatch --partition=gpu_a100 --cpus-per-task=18 jobs/bench_mig.sh --where snellius_a100
#
# sbatch flags on the command line override the #SBATCH lines below.
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:30:00
#SBATCH --job-name=dl1-bench
#SBATCH --output=slurm-%j.out

source jobs/_env.sh
"$PYTHON" bench.py "$@"
