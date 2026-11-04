# Seminar 1: the tooling

The training loop you wrote in assignment 1 does five jobs at once:

- it holds the hyperparameters;
- it moves data to the device;
- it runs the optimisation;
- it computes the metrics;
- it reports them.

All five grow as a project does. The configuration has to differ between your laptop and
the cluster. The device becomes several devices. The metrics have to stay correct when
each GPU sees a different part of the data. The reports have to outlive your terminal.

Four libraries take over the parts that are the same in every project:

| Job | Library | In this repo |
|---|---|---|
| Hyperparameters, per-machine settings | OmegaConf | `dl1/config.py`, `configs/` |
| The loop, devices, precision, checkpoints | Lightning | `dl1/module.py`, `train.py` |
| Metrics that are correct across batches and GPUs | torchmetrics | `dl1/metrics.py` |
| Logging, comparing runs | wandb | `train.py` (`WandbLogger`) |

None of them makes your run fast. What they buy you is that the fast settings, once
found, are applied the same way every time. That is what [seminar 2](../seminar_2_gpu/README.md)
builds on.

---

## 1. OmegaConf: one config, in layers

A config is a YAML file loaded into a `DictConfig`. You read it with dots
(`cfg.data.batch_size`). It can refer to other keys or to environment variables
(`${oc.env:TMPDIR,/tmp}`). It can be overridden from the command line with `key=value`.

### Layers, each with one job

`dl1/config.py` merges four layers, each overriding the one before:

```
configs/base.yaml           what is trained: data, model, optimiser
configs/where/<machine>     how to keep this machine busy: workers, precision, compile
--overlay <file.yaml>       an experiment on top (every configs/failures/ file is one)
key=value                   the command line, which wins over everything
```

The core of it is a few lines:

```python
cfg = OmegaConf.load("configs/base.yaml")
OmegaConf.set_struct(cfg, True)  # unknown keys are now an error
for layer in [f"configs/where/{where}.yaml", *overlays]:
    cfg = OmegaConf.merge(cfg, OmegaConf.load(layer))
cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
```

The `where` layer is picked for you. Inside a SLURM job it comes from the partition
(`gpu_mig` → `snellius_mig`). Anywhere else it is `local`. On a partition with no
profile, the run stops rather than guessing.

**Why layers.** Settings like `num_workers`, `precision` or `compile` depend on the
machine, not on the experiment. Put them in the experiment config and every experiment
has to repeat them. Then one day an experiment doesn't: a cluster run quietly trains
with laptop settings, and nothing tells you.

**The rule that keeps layers honest: every key has one owner.** `base.yaml` owns the
model and the optimiser. `where/` owns the machine settings. An experiment that restates
a key owned by `where/` silently overrides it for every machine.

> In continuous-flow-asr, 29 cluster runs inherited the laptop's model size and input
> resolution, because no layer above the base said otherwise. That was about 45k SBU,
> close to half the project's budget, spent on results that answered the wrong question.
> Every one of them would have been caught by reading the printed config before
> submitting.

### Struct mode: typos are errors

Without struct mode, `data.num_worker=8` (missing an `s`) adds a new key nobody reads.
The run trains with the default and you never find out. With struct mode it raises:

```bash
python train.py data.num_worker=8
# omegaconf.errors.ConfigKeyError: Key 'num_worker' is not in struct
```

`tests/test_config.py` loads every profile and every failure overlay, so a typo in a
YAML file fails the tests, not the run.

### Habits

- **`python train.py --print-config` before every submit.** It prints the merged config,
  exactly as the run will see it.
- **Log the resolved config to wandb** (`train.py` does this). Months later, the config
  of a run is the only reliable record of what it was. Your notes about it are not.
- **`cfg.profile`** records which `where/` profile and overlays built the config, so
  every run says what it ran under.

**Try it:** write `configs/where/<your-machine>.yaml` for your own laptop. Set
`num_workers` to your core count minus one, and `precision` to what your GPU supports.
Add it to `WHERE_BY_PARTITION` only if it is a cluster partition.

---

## 2. Lightning: the loop you no longer write

What you wrote in assignment 1, and where it lives now:

| Assignment 1 | Lightning |
|---|---|
| `for epoch ... for batch ...` | `Trainer.fit` |
| `model.to(device)`, `x.to(device)` | `accelerator="auto"`, done for you |
| `optimizer.zero_grad(); loss.backward(); optimizer.step()` | return the loss from `training_step` |
| `model.eval()` + `torch.no_grad()` for validation | `validation_step`, done for you |
| keep the best model by validation accuracy | `ModelCheckpoint(monitor="val/acc")`, then `trainer.test(ckpt_path="best")` |
| mixed precision, `GradScaler` | `precision="bf16-mixed"` |
| several GPUs | `devices=4, strategy="ddp"` |

You write the parts specific to your problem in two classes:
- a `LightningModule` (`dl1/module.py`): the model, `training_step`, `validation_step`, `configure_optimizers`;
- a `LightningDataModule` (`dl1/data.py`): download, split, dataloaders.

Everything else is `Trainer` arguments (`train.py`).

### Things worth knowing

- **Callbacks** are hooks into the loop (`on_train_batch_end`, `on_validation_epoch_end`, …).
  `dl1/callbacks.py` has three, each small enough to read in a minute:
  - `DataWaitMonitor` measures how long the loop waits for data (seminar 2 relies on it).
  - `Heartbeat` prints a line every 500 steps for cluster logs.
  - `StopAtStep` ends a run early without shrinking its LR schedule, so a short run
    matches a full one step for step.
- **`self.log` has a cost** in two ways:
  - `on_step=True` writes to the logger every `log_every_n_steps`.
  - `prog_bar=True` turns the value into a Python number at every progress-bar refresh.
    That waits for the GPU to finish: a sync. With the default refresh rate of 1, that is
    a sync every step. This repo refreshes every 50 steps (`logging.progress_every`).
    Seminar 2, card 3.
- **`torch.compile`:** use `model.compile()`, in place, not `model = torch.compile(model)`.
  The second wraps your module, and every key in the checkpoint then starts with
  `_orig_mod.`, so loading it into an uncompiled model fails.
- **On SLURM**, Lightning catches the signal sent at the walltime limit and exits
  cleanly. In wandb, a run killed by its walltime then looks finished. Check the step
  count, and ask SLURM to mail you (`--mail-type=TIME_LIMIT_90`, in `jobs/`).

---

## 3. torchmetrics: metrics are state, not numbers

In assignment 1, `evaluate_model` had a hint: *batches differ in size, so weight each
batch accuracy*. That is the whole idea behind torchmetrics. A metric keeps **sufficient
statistics** (sums and counts). It adds to them with every batch, and divides only once,
at the end:

```python
class PerClassAccuracy(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.add_state("correct", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, logits, target):
        hit = (logits.argmax(dim=-1) == target).float()
        self.correct.scatter_add_(0, target, hit)
        self.total.scatter_add_(0, target, torch.ones_like(hit))

    def compute(self):
        return self.correct / self.total
```

`tests/test_metrics.py::test_mean_of_batch_accuracies_is_not_the_accuracy` shows the
difference: four batches of 64, 64, 17 and 3, and the mean of their accuracies is not
the accuracy.

### Why this matters most on several GPUs

Under DDP each GPU runs its own process and sees its own shard of the data. With
`dist_reduce_fx="sum"`, `compute()` first sums every state across processes, so every
process gets the metric over the whole dataset. A mean of per-process accuracies is
wrong for the same reason a mean of per-batch accuracies is.

`tests/test_metric_ddp.py` checks this for real: two processes on the CPU (the `gloo`
backend), shards of 37 and 63 samples. It asserts that the synced metrics equal those of
one process that saw everything, and that the naive mean does not.

> DDP was never run in continuous-flow-asr, so there's no war story for this section.
> The test is the evidence: it runs on any laptop.

A few more things to know:
- **DistributedSampler pads each shard to the same length** by repeating a few samples,
  so on several GPUs a handful of validation samples count twice. The on-GPU pipeline
  here (`GpuBatches` in `dl1/data.py`) shards without padding, so its validation is exact.
- **Logging metrics.** `self.log("train/acc", self.train_acc, on_epoch=True)` hands the
  metric object to Lightning, which computes, syncs and resets it at the epoch's end. Or
  compute it yourself, as `_log_eval_epoch` does for the per-class vector. Then log the
  value without `sync_dist=True`: `compute()` already synced it.
- **Keep metrics on the GPU, and expensive ones out of the training step.** Both metrics
  in `dl1/metrics.py` update with tensor ops only: no `.item()`, no `.cpu()`, no Python
  loop over the batch.

> In continuous-flow-asr, a word error rate computed on the CPU at every training step
> took half of each step: 256.6 ms per step with it, 129.9 ms without. It now runs at
> validation only. Seminar 2, card 3.

**Try it:**
1. In `dl1/metrics.py`, change one `dist_reduce_fx="sum"` to `"mean"`, and run
   `uv run pytest tests/test_metric_ddp.py`. Explain why it fails, given uneven shards.
2. Write a `TopKAccuracy` metric, and test it against
   `torchmetrics.classification.MulticlassAccuracy(top_k=5)`, as the tests here do for ECE.

---

## 4. wandb: runs you can compare

`WandbLogger` sends everything logged with `self.log` to wandb, together with the
resolved config. It takes `group` and `tags`: give every run of a sweep the same group
(`scripts/run_parallel.py` does this), so they can be compared on one page.

### On Snellius

- **Compute nodes have no internet.** Run offline (`WANDB_MODE=offline`, which the job
  scripts set), then `wandb sync <run directory>` from a login node. Run `wandb login`
  once on a login node first.
- **Home is small.** wandb keeps a staging copy of everything it uploads. Point
  `WANDB_DIR` at scratch (the job scripts do).

> A full home quota once made `torch.save` fail in the middle of writing a checkpoint,
> and killed two runs at the very end.

### What to look at

- **The System panel** shows GPU utilisation, memory and power over time, where the
  driver reports them. Seminar 2 explains why utilisation is the least useful of the three.
- **`perf/samples_per_s` and `perf/data_wait_frac`** (from `DataWaitMonitor`) say how
  fast the run went and how much of that time it waited for data. Seminar 2 starts here.
- **`train/device/samples_per_sec`, `train/device/mfu`** come from Lightning's
  `ThroughputMonitor`, every `log_every_n_steps`. MFU (model FLOP utilisation) is the
  share of the GPU's peak FLOP/s your model achieves. It only appears when Lightning
  knows the GPU's peak. On several GPUs, the same keys without `device/` are totals for
  all of them.
