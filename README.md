# DL1 seminars: training tooling, and keeping the GPU busy

In assignment 1 you wrote a training loop by hand. This repository is what that loop
grows into once you need configs that change per machine, metrics that stay correct on
several GPUs, logs you can compare across runs, and a GPU that is actually busy. It is a
small but complete project: a ResNet-18 trained on
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), which you can run on your
laptop and on Snellius, and store as boilerplate for your own projects.

There are two seminars:

1. **[The tooling](seminar_1_tooling/README.md).** [OmegaConf](https://omegaconf.readthedocs.io)
   for configs, [Lightning](https://lightning.ai/docs/pytorch/stable/) for the training
   loop, [torchmetrics](https://lightning.ai/docs/torchmetrics/stable/) for metrics, and
   [wandb](https://docs.wandb.ai) for logging: what each one takes off your hands and
   where each one bites.
2. **[Keeping the GPU busy](seminar_2_gpu/README.md).** A cookbook: how to find out
   *why* a run is slow, then one card per bottleneck with a config that reproduces it,
   the fix, and what it cost in a real project. The real project is in
   [war stories](seminar_2_gpu/war_stories.md).

## Setting up

As in assignment 1, the environment is managed with [uv](https://docs.astral.sh/uv/)
and pinned in `uv.lock`:

```bash
uv sync                               # CUDA build of PyTorch (Snellius, NVIDIA laptops)
uv sync --no-group cuda --group cpu   # CPU-only build (no NVIDIA GPU)
uv run pytest                         # the tests, including a 2-process DDP test on CPU
```

The same caveats as in assignment 1 apply. A bare `uv run` after a CPU sync reinstalls
the CUDA build, so either repeat the flags or use `uv run --no-sync`. A CPU build on a
GPU machine does not raise an error, it just trains on the CPU. `train.py` prints the
device it found as its first line, and the Snellius job scripts refuse to start without
a GPU.

The first run downloads CIFAR-10 (170 MB) into `data/`.

## Running

```bash
python train.py --print-config                 # what would run, and exit
python train.py                                # train with the profile for this machine
python train.py data.batch_size=256 train.max_epochs=5
python train.py --overlay configs/failures/01_cpu_bound_loader.yaml
python bench.py                                # is this run input-bound or compute-bound?
python scripts/run_parallel.py --grid seed=0,1 # two jobs sharing one GPU
```

On Snellius, submit the job scripts instead of running on a login node:

```bash
sbatch jobs/train_mig.sh                       # one MIG slice of an A100: the default
sbatch jobs/bench_mig.sh                       # the benchmark, on a slice
sbatch jobs/train_a100.sh                      # a full A100, when you can fill it
```

Compute nodes have no internet, so wandb runs offline there. Afterwards, upload the run
from a login node with `wandb sync <the directory the job printed>`.

## What is where

```
train.py                 entry point: config -> data + model -> Lightning Trainer
bench.py                 times the step three ways: real, replayed batch, loader only
configs/
  base.yaml              what is trained (data, model, optimiser)
  where/                 how to keep each machine busy: local, snellius_mig, snellius_a100
  failures/              one overlay per cookbook card, each reproducing one bottleneck
dl1/
  config.py              layered config loading (OmegaConf), with typo-proof struct mode
  data.py                CIFAR-10 three ways: classic workers, all-on-GPU, one PNG per image
  models.py              CIFAR ResNet-18 and a small MLP
  metrics.py             two custom torchmetrics metrics, correct under DDP
  module.py              the LightningModule: loss, metrics, logging, optimiser
  callbacks.py           data-wait monitor, heartbeat, stop-at-step
  prefetch.py            overlap the next batch's copy to the GPU with the current step
scripts/
  run_parallel.py        a grid of jobs, N at a time, on one machine
  export_pngs.py         CIFAR-10 as 50,000 files, for the small-files card
  bench_reads.py         how fast a filesystem hands out files, by processes x threads
jobs/                    SLURM scripts for Snellius (gpu_mig, gpu_a100)
tests/                   config merging, metrics against torchmetrics, metrics under DDP
```

To reuse this for your own project, the pieces that carry over unchanged are
`dl1/config.py`, `dl1/callbacks.py`, `dl1/prefetch.py`, `bench.py`,
`scripts/run_parallel.py` and `jobs/`. `dl1/metrics.py` is the template for your own
metrics. The data module and the model are the parts you replace.

## Code style

[Ruff](https://docs.astral.sh/ruff/) is set up as in assignment 1 and runs on every push:

```bash
uv run ruff check .
uv run ruff format .
```
