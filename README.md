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
`uv run --no-sync`. The first run downloads CIFAR-10 from the Hugging Face Hub
([`uoft-cs/cifar10`](https://huggingface.co/datasets/uoft-cs/cifar10), ~130 MB) into `data/`.

## Running

Run everything from the root of the repository:

```bash
python train.py --print-config                   # the merged config, without training
python train.py                                  # train, with the settings for this machine
python train.py optim.lr=0.05 train.max_epochs=5 # override any setting with key=value
python train.py --overlay configs/experiments/resnet18.yaml  # an experiment on top
```

## Playing with the config

`--print-config` builds the config and prints it without training, so each of these takes a
second. The layers merge in order (`src/config.py`), and each one overrides the one before:

```bash
# Follow one key through the four layers.
python train.py --print-config | grep "lr:"                               # 0.1  base.yaml
python train.py --print-config --where snellius_mig | grep "lr:"          # 0.1  + where/: a machine
                                                                          #      does not change what is trained
python train.py --print-config --where snellius_mig \
    --overlay configs/experiments/large_batch.yaml | grep "lr:"           # 0.4  + an experiment
python train.py --print-config --where snellius_mig \
    --overlay configs/experiments/large_batch.yaml optim.lr=0.2 | grep "lr:"  # 0.2  + the command line

# Experiments stack, in order: ResNet-18, for 1 epoch, tagged debug.
python train.py --print-config --overlay configs/experiments/resnet18.yaml \
    --overlay configs/experiments/debug.yaml

# A typo is an error, not a silently ignored key (struct mode).
python train.py --print-config data.num_worker=8
# ConfigKeyError: Key 'num_worker' is not in struct. Did you mean: 'num_workers'?

# Interpolation: ${oc.env:USER} fills in your username on Snellius.
python train.py --print-config --where snellius_mig | grep scratch-shared

# The config records which layers built it, and wandb stores it with the run.
python train.py --print-config --overlay configs/experiments/large_batch.yaml | grep -A3 "^profile:"
```

Struct mode catches unknown keys, not wrong values: `data.augment=flase` is accepted as the
string `"flase"`. To ask a question of your own, write a file in `configs/experiments/` with only
the keys it changes (`data.augment: false`, say), and pass it with `--overlay`.

## On Snellius

Train inside a SLURM job, never on a login node. From the root of the repository:

```bash
sbatch jobs/install_env.sh        # once: builds the conda environment uvadlc_seminars
sbatch jobs/download_data.sh      # once: CIFAR-10 into /scratch-shared/$USER/data (CPU job)
module load 2025 Anaconda3/2025.06-1 && source activate uvadlc_seminars && wandb login  # once
sbatch jobs/train.sh              # then, as often as you like, with any key=value overrides
```

The data lives in `/scratch-shared/$USER/data`. Runs and checkpoints go
to `/scratch-shared/$USER/outputs`, because home is small.

## What is where

```text
notebooks/prototyping.ipynb   the project in plain PyTorch
train.py                      entry point: config -> data + model -> Lightning Trainer
configs/
  base.yaml                   what is trained: data, model, optimiser
  where/                      per-machine settings: local, snellius_mig, snellius_a100
  experiments/                one question each, as overlays: debug, wide_cnn, resnet18, ...
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
