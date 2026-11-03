# War stories

Everything in the cookbook was learned the expensive way in one research project: a
speech recogniser trained from scratch on precomputed features (each training sample is
an audio clip already passed through a frozen encoder, stored on disk). It was developed
on a laptop with an RTX 3060 (6 GB) under WSL2, and trained on Snellius A100s and H100s,
on a fixed budget of 100k SBU.

Every number below was measured in that project. The model and data are different from
CIFAR-10, so the numbers themselves won't carry over. The causes do.

---

## The step that got 3.7× faster without touching the model

One A100, batch 128, each row adding one change to the previous one:

| Change | ms per step |
|---|---|
| As every run so far had been | 256.6 |
| Training-time word error rate off (a CPU metric, card 3) | 129.9 |
| `torch.compile` (card 6) | 98.7 |
| bf16 mixed precision (card 5) | 70.0 |

Before this table existed, 70 cluster runs had been trained at the first row's speed,
with the GPU ~37% busy for every hour they were billed.

The first row is the lesson. The slowest thing in the step was not a layer, it was a
metric. Decoding transcripts and scoring them on the CPU at every step left the GPU
idle for half of each step. Nobody had noticed, because the metric was useful, and the
training curve looked fine.

## 29 runs on the wrong model

A sweep varied a few hyperparameters and left everything else to the defaults. The
defaults lived in `base.yaml`, and `base.yaml` was written for the laptop: a tiny model,
and inputs downsampled 4× to fit a 6 GB GPU. Twenty-nine cluster runs trained that
laptop model. The results were plausible, and answered a question nobody had asked.
About 45k SBU.

What changed afterwards:
- Machine settings moved into profiles (`configs/where/` in this repo).
- The cluster refused to train any config that wasn't a named recipe stating the model
  size explicitly.
- `--print-config` became the step before every submission.

The same project had a smaller version of the same mistake. An experiment file pinned
`num_epochs: 50`, which silently beat the profile's 300. The sweep was priced, and its
walltime set, for 300.

## Ten runs killed at the finish line

Walltimes were estimated from a guess of 43 s per epoch. The measured value was
79–108 s. Ten runs hit their 6-hour limit between epochs 198 and 272 of 300.
- They used a one-cycle learning-rate schedule, so at the cut-off the LR was still four
  orders of magnitude above its final value. The checkpoints were useless.
- 15.4k SBU, half of that month's spend.
- In wandb all ten showed as **finished**: Lightning catches SLURM's warning signal and
  exits cleanly.

What changed afterwards:
- Walltimes are derived from measured steps per second, with a margin. Billing is on
  elapsed time, so a generous walltime costs nothing.
- Jobs mail at 90% of their walltime.
- Short screening runs (100 epochs, one seed) rank the options first. Their ranking
  matched the full runs' exactly, at about a third of the cost.

## The memory limit that made reads 3× slower

The training features were 102 GiB of memory-mapped files on GPFS, Snellius's shared
filesystem. Linux caches file pages in RAM, but on Snellius that cache counts against
the job's `--mem`.
- At `--mem=64G` the cache couldn't hold the data, so every epoch re-read all of it over
  the network: 219 samples/s, against 657 once it fit. That is a 3× longer walltime from
  one SLURM flag.
- At `120G` it fit, but the cache plus 16 worker processes went over the limit. The
  kernel killed a worker mid-epoch, *sometimes*: cached pages can be reclaimed, so
  whether the limit was hit depended on timing.
- The budget that worked: data + workers + main process, 160G.

Then the dataset grew to 1.1 TB, which no node's RAM can cache. Reads off GPFS were
latency-bound. Throughput grew with the number of reads in flight, from 46 samples/s at
16 processes to 215 at 128, and then flattened at about 0.8 GB/s, the node's share of
the filesystem. The fix was to copy the data to the node's local NVMe at the start of
each job (16 parallel copies of ~10 GB files), which gave 765 samples/s.

## Benchmarks that lied

Each of these produced a confident, wrong number:

- **Trial order.** In a batch-size benchmark on GPFS, the time per step *fell* as the
  batch grew, which is impossible. Each trial read the same files, warmer in the cache
  than the trial before. Fix: read everything once before timing, and keep the best of
  several repeats.
- **A login-node benchmark.** Plain positioned reads measured 1.75× faster than memory
  mapping on a login node, and were adopted. On a compute node they turned out to bypass
  the page cache entirely: the third epoch was as slow as the first (137 samples/s,
  against 9,632 for memory-mapped reads once cached). Reverted two days later.
- **Everything resident at once.** A benchmark kept all six batch sizes' tensors alive:
  7.57 GiB on a 6 GB card, spilling to host memory. Throughput came out flat and the step
  time perfectly linear in the batch size. `bench.py` runs each row in a fresh process
  for this reason.
- **Laptop thermals.** A laptop's clock speed drifts as it heats up, which skewed the
  first pass of a benchmark against the later ones. Interleave the trials and take
  medians.
- **Timing setup.** Timing all of `fit` spreads worker start-up, compilation and the
  first allocations over the few timed steps. Time only after a warmup, and synchronise
  with the GPU at both ends (`bench.py`'s `StepTimer`).

## A fix that made it slower

A run on an H100 seemed capped by the DataLoader's single thread that pins memory, so it
got a background thread pinning batches in parallel. The result: 490 samples/s, against
741 before. The new thread competed with the training process for the CPU (and for
Python's GIL), and the training process is the one that launches the GPU's work.
Reverted. Measure every fix, including the obvious ones.

## Cluster pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `torch.save` fails half-way through a checkpoint | home quota full (checkpoints, wandb's staging copies, pip/uv caches) | `myquota`; caches and resume files on scratch |
| data you put on scratch is gone | `/scratch-shared` purges old files | never keep the only copy of anything there |
| `TypeError: unsupported operand \|` at submit time | the login node's system Python 3.9 | activate the environment first |
| a job runs last week's code | the cluster checkout was never pulled | print the commit at job start (`jobs/_env.sh`) |
| a job sits at a login prompt until its walltime | wandb not logged in on the cluster | `wandb login` once on a login node; offline mode on compute nodes |
| a run looks finished at epoch 201 of 300 | walltime kill, caught cleanly by Lightning | `--mail-type=TIME_LIMIT_90`; compare the step count with the plan |
| `nvidia-smi -i 0` in a 2-GPU job reports someone else's GPU; `srun --overlap nvidia-smi` shows none of yours | `nvidia-smi` numbers the node's GPUs, ignoring `CUDA_VISIBLE_DEVICES` | read utilisation from inside the job (wandb's system metrics, `DataWaitMonitor`) |
| an overnight laptop run crashes WSL | two jobs outgrew WSL's memory limit | raise it in `.wslconfig` |
