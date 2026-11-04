# DL1 coding seminars

One small project, built twice: an image classifier on
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), first as a plain PyTorch notebook, then
as a project that uses [Lightning](https://lightning.ai/docs/pytorch/stable/),
[wandb](https://docs.wandb.ai), [torchmetrics](https://lightning.ai/docs/torchmetrics/stable/)
and [OmegaConf](https://omegaconf.readthedocs.io). The later seminars build on the same layout.

## How to go through it

1. **`notebooks/prototyping.ipynb`.** The whole project in plain PyTorch. Lines marked `# ->`
   are boilerplate. Each says which library takes it over, and where it ends up in the project.
2. **`train.py`.** The entry point. It builds the config, the data and the model, and hands
   them to a Lightning `Trainer`.
3. **`src/`.** The pieces that `train.py` puts together. Read them in this order: `data.py`,
   `pl_module.py`, `metrics.py`, `config.py`. The settings are in `configs/`.

## Setting up

The environment is managed with [uv](https://docs.astral.sh/uv/), as in assignment 1:

```bash
uv sync                               # NVIDIA GPU (Snellius, most laptops), or a Mac
uv sync --no-group cuda --group cpu   # no GPU at all
uv run wandb login                    # once, to log runs to your wandb account
```

After a CPU-only sync, a bare `uv run` reinstalls the CUDA build: repeat the flags, or use
`uv run --no-sync`. The first run downloads CIFAR-10 (170 MB) into `data/`.

## Running

Run everything from the root of the repository:

```bash
python train.py --print-config                   # the merged config, without training
python train.py                                  # train, with the settings for this machine
python train.py optim.lr=0.05 train.max_epochs=5 # override any setting with key=value
python train.py --overlay my_experiment.yaml     # an experiment layered on top
```

## On Snellius

Train inside a SLURM job, never on a login node. From the root of the repository:

```bash
sbatch jobs/install_env.sh        # once: builds the conda environment uvadlc_seminars
module load 2025 Anaconda3/2025.06-1 && source activate uvadlc_seminars && wandb login  # once
sbatch jobs/train.sh              # then, as often as you like, with any key=value overrides
```

The first job downloads CIFAR-10 into `/scratch-shared/$USER/data`. Runs and checkpoints go
to `/scratch-shared/$USER/outputs`, because home is small.

## What is where

```text
notebooks/prototyping.ipynb   the project in plain PyTorch
train.py                      entry point: config -> data + model -> Lightning Trainer
configs/
  base.yaml                   what is trained: data, model, optimiser
  where/                      per-machine settings: local, snellius_mig, snellius_a100
src/
  data.py                     the CIFAR-10 LightningDataModule
  models.py                   the models: SmallCNN (default), ResNet-18, MLP
  pl_module.py                the LightningModule: loss, metrics, logging, optimiser
  metrics.py                  a custom torchmetrics metric, correct across GPUs
  config.py                   loads and merges the config layers (OmegaConf)
```

## Code style

[Ruff](https://docs.astral.sh/ruff/) checks every push:

```bash
uv run ruff check .
uv run ruff format .
```
