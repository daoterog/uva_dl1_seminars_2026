# DL1 seminars: training tooling

In assignment 1 you wrote a training loop by hand. This repository is what that loop
grows into once you need configs that change per machine, metrics that stay correct on
several GPUs, and logs you can compare across runs. It is a small but complete project:
a ResNet-18 trained on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), which you
can run on your laptop and on Snellius, and store as boilerplate for your own projects.

**[The tooling](seminar_1_tooling/README.md).** [Lightning](https://lightning.ai/docs/pytorch/stable/)
for the training loop, [wandb](https://docs.wandb.ai) for logging,
[torchmetrics](https://lightning.ai/docs/torchmetrics/stable/) for metrics, and
[OmegaConf](https://omegaconf.readthedocs.io) for configs: what each one takes off your
hands and where each one bites.

## Setting up

As in assignment 1, the environment is managed with [uv](https://docs.astral.sh/uv/)
and pinned in `uv.lock`:

```bash
uv sync                               # CUDA build of PyTorch (Snellius, NVIDIA laptops)
uv sync --no-group cuda --group cpu   # CPU-only build (no NVIDIA GPU)
```

The same caveats as in assignment 1 apply. A bare `uv run` after a CPU sync reinstalls
the CUDA build, so either repeat the flags or use `uv run --no-sync`. A CPU build on a
GPU machine does not raise an error, it just trains on the CPU. `train.py` prints the
device it found as its first line.

The first run downloads CIFAR-10 (170 MB) into `data/`.

## Running

```bash
python train.py --print-config                 # what would run, and exit
python train.py                                # train with the profile for this machine
python train.py data.batch_size=256 train.max_epochs=5
python train.py --overlay my_experiment.yaml   # an experiment layered on top
```

On Snellius, run inside a SLURM job, never on a login node. Compute nodes have no
internet, so wandb runs offline there. Afterwards, upload the run from a login node with
`wandb sync <the directory the job printed>`.

## What is where

```
notebooks/
  prototyping.ipynb      the same project in plain PyTorch: where every project starts
train.py                 entry point: config -> data + model -> Lightning Trainer
configs/
  base.yaml              what is trained (data, model, optimiser)
  where/                 how to keep each machine busy: local, snellius_mig, snellius_a100
src/
  config.py              layered config loading (OmegaConf), with typo-proof struct mode
  data.py                the CIFAR-10 LightningDataModule
  models.py              CIFAR ResNet-18 and a small MLP
  metrics.py             two custom torchmetrics metrics, correct under DDP
  pl_module.py           the LightningModule: loss, metrics, logging, optimiser
```

To reuse this for your own project, `src/config.py` carries over unchanged and
`src/metrics.py` is the template for your own metrics. The data module and the model are
the parts you replace.

## Code style

[Ruff](https://docs.astral.sh/ruff/) is set up as in assignment 1 and runs on every push:

```bash
uv run ruff check .
uv run ruff format .
```
