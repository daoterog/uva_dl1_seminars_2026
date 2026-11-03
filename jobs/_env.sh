# Sourced by every job script: one place for the environment, so the scripts stay short.
#
# What it guards against, each of which has cost someone a job:
# - running a checkout you forgot to pull: the commit is printed first thing;
# - a CPU-only torch on a GPU node: the device check fails the job in seconds, not hours;
# - a full home quota: wandb's files go to scratch.

# Load the modules your course environment needs here (see the Snellius docs for the
# current module year), or rely on uv: the .venv below carries its own Python. Before
# `set -u`, which the module command does not survive.
module purge

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

echo "commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown) | host $(hostname) |" \
     "job ${SLURM_JOB_ID:-local} | partition ${SLURM_JOB_PARTITION:-none} |" \
     "cpus ${SLURM_CPUS_PER_TASK:-?}"
nvidia-smi -L || true

PYTHON="${PWD}/.venv/bin/python"
"$PYTHON" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else \
'torch sees no GPU: did uv sync install the CPU build? (uv sync, without --group cpu)')"

# Compute nodes have no internet: wandb runs offline and is synced afterwards from a
# login node with `wandb sync <dir>`. Keep its files off the small home quota.
export WANDB_MODE=offline
export WANDB_DIR="${TMPDIR:-/tmp}/wandb"
mkdir -p "$WANDB_DIR"
