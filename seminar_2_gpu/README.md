# Seminar 2: keeping the GPU busy

A GPU you were allocated costs the same whether it computes or waits. On Snellius that
cost is your budget; on your laptop it is your afternoon. In the project behind this
seminar ([war stories](war_stories.md)), the same training step on the same A100 went
from 256.6 ms to 70.0 ms, **3.7× faster, without changing the model**. None of the time
was lost inside the GPU's kernels. It went to metrics computed on the CPU, data
loading, precision settings, and memory limits.

There is no single fix, because there is no single cause. This seminar is a cookbook:
first **how to find out what your run is waiting for**, then one card per cause, each
with a config that reproduces it in this repo, the fix, and what it cost in a real
project.

---

## How a training step spends its time

The CPU and the GPU work **asynchronously**. Your Python code on the CPU does not
compute anything on the GPU; it *queues* work (kernel launches) and moves on. The GPU
works through the queue. As long as the CPU stays ahead, the GPU never waits.

Every step has to get through several stages, and the slowest one sets the pace:

```
 disk / network FS ──> CPU workers ──> host RAM ──> copy to GPU ──> GPU kernels
   (read files)       (decode, augment)  (pinned?)   (PCIe)        (forward, backward)
                                                         ▲
                                       Python in the main process: launches every kernel,
                                       runs your metrics and logging, and can stall the
                                       queue with a sync
```

A **sync** is any line that needs a GPU result on the CPU:
- `loss.item()`, `float(t)`, `print(t)`;
- `t.cpu()`, `t.numpy()`;
- `if t > 0:` on a GPU tensor;
- `torch.cuda.synchronize()`.

At a sync the CPU stops and waits until the GPU has finished everything queued so far,
so it can no longer queue ahead. One per epoch is free. One per step can cost you a
large share of the run.

---

## Diagnose first

Never start with a fix. Every fix below helps one cause and does nothing (or harm) for
the others. In the real project, `torch.compile` measured **neutral** while the run was
waiting for data, and **+33%** once the data loading was fixed.

### 0. Is it on the GPU at all?

`train.py` prints the device as its first line. If it says `CPU only` on a machine with
a GPU, you installed the CPU build of PyTorch (see the root README). Nothing raises an
error. The run is just a hundred times slower.

### 1. Measure throughput, not "GPU utilisation"

`nvidia-smi`'s *GPU-Util* is the fraction of time **at least one kernel** was running.
It doesn't say how much of the GPU that kernel used. A model that keeps one tiny kernel
running at all times reads 100% while using a few percent of the chip. On a **MIG
slice**, `nvidia-smi` does not show utilisation for your slice at all.

Measure what you actually want instead:
- **samples per second**, which is what makes the run finish sooner;
- **how much of the time the loop waits for its data**.

`DataWaitMonitor` (`dl1/callbacks.py`) logs both every epoch, as `perf/samples_per_s`
and `perf/data_wait_frac`. It runs in the training process, so it works on a laptop, a
MIG slice and a full A100 alike.

### 2. Real vs replay vs loader: `bench.py`

```bash
python bench.py                       # or: sbatch jobs/bench_mig.sh
```

```
row       batch   ms/step  samples/s   wait  peak GiB
real        128       ...        ...    ...       ...
replay      128       ...        ...    ...       ...
loader      128       ...        ...      -         -
```

- **real** is your training loop as it is.
- **replay** feeds the same loop one batch that is already on the GPU, over and over. No
  loader, no copy. It is the fastest this model can train at this batch size on this GPU.
- **loader** runs your data pipeline alone, with no model. It is the fastest your input
  can arrive.

| What you see | What it means | Go to |
|---|---|---|
| replay ≫ real, loader ≈ real | the input pipeline is the limit | cards 1, 2, 8, 10 |
| replay ≫ real, loader ≫ real | each is fast alone, slow together: they compete for CPU (workers vs the main process), or the copy is not overlapped | cards 1, 2, 3 |
| replay ≈ real, both slow | the GPU step itself | cards 3, 4, 5, 6 |
| replay ≈ real, fast, and still the run takes forever | time goes outside the training step | cards 7, 8 |
| `failed` | out of memory | card 9 |

> In continuous-flow-asr, on an A100: real 525 samples/s, replay 742. So ~30% of the
> time went to moving the batch to the GPU (card 2). On an H100: GPU alone 1,341, loader
> alone 1,043, both together only 741. They were fighting over the CPU.

### 3. Batch-size ladder

```bash
python bench.py --rows real,replay --ladder 32,64,128,256,512,1024
```

Throughput rises with the batch size, then flattens. **Take the smallest batch within
~5% of the best.** A bigger batch changes the optimisation too: the learning rate you
tuned belongs to the batch you tuned it at.

> A100, real loop: batch 32 → 457 samples/s, 64 → 500, 128 → 525, 256 → 545, 512 out of
> memory. 128 is 96% of the best, and the batch the learning rate was tuned at.

### 4. Profile, once you know where to look

- `python train.py train.profiler=simple train.max_steps=200 --no-test` prints the time
  spent in each Lightning hook.
- `train.profiler=pytorch` writes a trace you can open in [Perfetto](https://ui.perfetto.dev).
  Look for gaps in the GPU's row, and for `cudaStreamSynchronize` or `aten::item` on the
  CPU's row. Each one is a sync.
- Set `TORCH_LOGS=recompiles` to see every time `torch.compile` compiles again (card 6).

(The profilers were not used in continuous-flow-asr; everything there was found with
benchmarks like `bench.py`. They are the next step when a benchmark says "the step is
slow" and you need to know which line.)

### 5. How close to the hardware? MFU

Lightning's `ThroughputMonitor` logs `train/device/mfu`: FLOPs your model does per
second, divided by the GPU's peak. FLOPs per step are counted once, on the meta device
(`Classifier._count_flops`). The metric only appears for GPUs whose peak Lightning
knows. On a MIG slice, compare against the slice's share of the card's peak: Lightning
may report the whole card's, or nothing. For small CIFAR models, a few tens of percent
is a good result.

---

## The cards

Each card:
- **Symptom:** what you notice.
- **Reproduce:** a config overlay in `configs/failures/`, to compare with and without.
- **Fix.**
- **In continuous-flow-asr:** the measured cost in a real project.

Run any card with `bench.py` (fast, no training) or `train.py` (watch `perf/*` in wandb):

```bash
python bench.py --overlay configs/failures/01_cpu_bound_loader.yaml   # the failure
python bench.py                                                       # your profile
```

### Card 1: the CPU can't prepare batches fast enough

**Symptom.** High `perf/data_wait_frac`. `bench.py`: replay much faster than real, loader ≈ real.
CPU cores at 100% in `htop`, GPU mostly idle.

**Reproduce.** `01_cpu_bound_loader.yaml`: PIL transforms in the main process, no workers.

**Fix.**
- **Workers = cores − 1.** One core stays free for the main process, which launches the
  GPU's work. On Snellius, cores are whatever you asked `--cpus-per-task` for, not what
  the node has.
- **Do less per sample on the CPU.** Here, `data.pipeline=gpu` keeps the whole dataset
  on the GPU and augments it there, so no CPU is involved at all. It is only possible
  because CIFAR-10 fits in GPU memory (150 MB); when your dataset does, it is the
  fastest option there is.
- **Move fewer bytes between processes.** Workers hand their batches to the main process
  through shared memory. Send the smallest dtype you can (uint8 images, fp16 features)
  and convert on the GPU.

**In continuous-flow-asr.**
- Reducing features in fp16 on the CPU, which has no fast fp16 path, took the loader
  from 106 to 47 ms per batch once done in fp32 (2.3×).
- Shipping fp16 instead of fp32 from workers took batch 256 from 1,043 to 2,656 samples/s
  (2.5×). Batch 512 had crashed with a bus error, `/dev/shm` full, before that.
- With 18 cores, the 17th worker was worth +12% (711 → 798 samples/s).

### Card 2: the copy to the GPU sits on the critical path

**Symptom.** replay faster than real, while the loader alone keeps up. Large batches.

**Reproduce.** `02_pageable_copy.yaml`: unpinned memory, an MLP at batch 2048, so the
copy is large next to the compute.

**Fix.**
- **`pin_memory=True`.** A copy from pageable memory is always synchronous, and it goes
  through an extra staging buffer.
- **Overlap the copy with compute** (`data.gpu_prefetch=true`, `dl1/prefetch.py`). The
  next batch's copy runs on a second CUDA stream while the current batch trains.

**Don't over-apply it.** With ResNet-18 at batch 128, a batch is 1.5 MB. Its copy takes
well under a millisecond, and this card will show you nothing. Fix what you measured.

**In continuous-flow-asr.** Batches were 490 MB.
- The copy was ~30% of the step: 244 ms vs 172.5 ms with the batch already on the GPU.
- Prefetching on a side stream: 643 → 736 samples/s, with bit-identical batches.

### Card 3: the CPU waits for the GPU every step

**Symptom.** replay ≈ real, and both slower than they should be, especially for small
models. In a profiler trace, a `cudaStreamSynchronize` every step.

**Reproduce.** `03_host_syncs.yaml`, which turns on every step:
- `float(loss)`;
- a NumPy accuracy on CPU copies of the predictions;
- `if not torch.isfinite(loss)`;
- a progress bar refreshing every step.

**Fix.**
- Metrics as torchmetrics objects, updated on the GPU and logged per epoch (`dl1/module.py`).
- Nothing per step that needs a Python number.
- Progress bar and logging on a cadence (`logging.progress_every`, `log_every_n_steps`).
- Metrics that genuinely need the CPU (text decoding, external scorers): validation only.

**In continuous-flow-asr.**
- **The biggest single win of the project.** A word error rate decoded and scored on the
  CPU at every training step: 68.6 ms of CPU per step, against 0.5 ms without it. Step
  time 256.6 → 129.9 ms (2×). The train/validation gap was still visible from the loss.
- A metric that updated one row at a time (`errors[b] += ...`, two host-device round
  trips per sample): 16.9 → 9.2 ms per batch, with one tensor add per batch.

### Card 4: too little work per step

**Symptom.** A small model or a small batch. replay ≈ real, GPU memory nearly empty,
throughput grows almost linearly with batch size. The GPU spends its time waiting for
the next kernel launch from Python.

**Reproduce.** `04_launch_bound.yaml`: an MLP at batch 32, fed from the GPU.

**Fix.** In this order:
1. **Batch ladder** (diagnose step 3).
2. **`train.compile=true`**: fewer, larger kernels. `train.compile_mode=reduce-overhead`
   also records the step as a CUDA graph and replays it with one launch.
3. **Two jobs on one GPU** (see [Two jobs, one GPU](#two-jobs-one-gpu)).
4. **A smaller GPU.** If a job can't fill an A100, run it on a MIG slice: you are billed
   for the slice.

**In continuous-flow-asr.**
- Local sweeps left a laptop GPU 30% busy with one job, and 59% with two.
- `torch.compile` at batch 16 bought nothing, for 45 s of compilation. A synthetic loop
  at batch 64 was 64% faster compiled, but the real loop was not, because it was waiting
  for data. Fix card 1 before card 4.

### Card 5: fp32 on hardware built for less

**Symptom.** Compute-bound (replay ≈ real) on an Ampere or newer GPU (A100, H100, RTX 30xx
and later), running `32-true`.

**Reproduce.** `05_full_precision.yaml`. Compare it with the profile's `bf16-mixed`.

**Fix.**
- **`train.precision=bf16-mixed`.** Matmuls and convolutions in bfloat16, weights and
  the optimiser in fp32. Unlike fp16, bf16 needs no loss scaling.
- **TF32 (`train.matmul_precision=high`)** for the matmuls autocast leaves in fp32.
  Measure it.
- GPUs without bf16 (T4, V100, GTX): `16-mixed`.

**In continuous-flow-asr.**
- bf16-mixed: 98.7 → 70.0 ms per step on the A100.
- TF32 gave +6% on the A100 (711 → 755 samples/s), because one layer stayed in fp32.
  On the laptop, under fp16, it made steps *slower* (38.5 → 43.4 ms).
- Results under different precisions are not comparable to the last decimal: compare
  bf16 runs with bf16 runs.

### Card 6: `torch.compile` that costs more than it gives

**Symptom.** A slow start. Long pauses at the start of validation, or at the end of
every epoch. No speed-up at all.

**Reproduce.** `06_compile_recompiles.yaml`, run with `TORCH_LOGS=recompiles`. It
compiles, and keeps a short last batch in training and validation. Each new shape is a
new compilation.

**Fix.**
- `drop_last=True` for training.
- For validation, accept one extra compilation, or pad the last batch.
- Compile only runs long enough to earn the warmup back (the laptop profile keeps it off).
- Compile in place (`model.compile()`) so checkpoints keep their keys (seminar 1).

**In continuous-flow-asr.**
- On the A100, compile took 129.9 → 98.7 ms per step. Locally it was 45 s of warmup for
  nothing.
- The short last validation batch cost a 53 s recompilation in every run.
- A model that indexed with data-dependent shapes (`nonzero`) broke the graph, and is
  trained uncompiled.

### Card 7: evaluation on the training clock

**Symptom.** Fast training epochs, slow run. Long gaps in `perf/samples_per_s` around
validation.

**Reproduce.** `07_eval_overhead.yaml`: validation every epoch, at batch 32, with no workers.

**Fix.**
- **Validate on a cadence** (`train.check_val_every_n_epoch`), sized so evaluation is a
  few percent of the run.
- **Evaluate at a large batch** (`data.eval_batch_size`). Under `no_grad` there are no
  activations to keep, so evaluation fits several times the training batch.

**In continuous-flow-asr.**
- Validation plus a sampling probe took 33 s against 54 s of training per epoch.
  Validating every 10 epochs brought evaluation to ~6% of the run.
- Evaluation at batch 512 instead of 32 was 24% cheaper per sample.

### Card 8: tiny datasets, endless epochs

**Symptom.** A small subset (an overfitting test, a debugging run), and far more
wall-clock time than steps × step time.

**Reproduce.** `08_tiny_dataset.yaml`: 512 images at batch 128 is 4 steps per epoch, for
500 epochs. Every epoch boundary restarts the loader's workers and runs the epoch hooks.

**Fix.** `data.steps_per_epoch=200`: each "epoch" is a fixed number of steps sampled
with replacement. Keep `persistent_workers=true`.

**In continuous-flow-asr.** 64 samples at batch 16 meant 6,400 epoch boundaries at
~106 ms each: about 11 minutes of overhead per run.

### Card 9: out of memory, GPU or host

**Symptom.**
- `CUDA out of memory`.
- A worker "killed by signal". On SLURM, `oom-kill` in the job log, sometimes and not
  every time.

**Reproduce.** `09_host_memory.yaml`: 16 workers with 8 prefetched batches each, and 16
persistent eval workers. Watch `free -g`, or `sstat -j <jobid> --format=MaxRSS` on
Snellius.

**Fix.**
- **Host RAM:**
  - Every worker is a full copy of the main process, and holds `prefetch_factor`
    batches; `prefetch_factor=1` is usually enough.
  - Evaluation needs few workers.
  - On a cluster, request `--mem` for the data, the workers and the page cache (card 10).
- **GPU memory:**
  - A profile written for a 40 GB card runs out on a 20 GB slice. Check `eval_batch_size`
    especially.
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps when the error says a lot is
    "reserved but unallocated" (fragmentation).
  - `peak GiB` in `bench.py` says how close you are.

**In continuous-flow-asr.**
- Prefetch depth 2 with 17 workers held ~17 GB of batches in flight per run, and got a
  job OOM-killed. Depth 1 ran at the same speed (711 vs 712 samples/s).
- Persistent eval workers meant 34 extra processes, and a 53 GB main process.
- On a MIG slice, a run inherited the A100 profile's eval batch of 512 and ran out of
  memory.

### Card 10: the filesystem

**Symptom.** Slow on the cluster, fast on your laptop. Fast in the second epoch, slow in
the first, or slow in all of them.

**Reproduce.** Write CIFAR-10 as 50,000 PNG files to the filesystem you want to test,
then train from them:

```bash
python scripts/export_pngs.py --out /scratch-shared/$USER/cifar10_png
python bench.py --overlay configs/failures/10_png_files.yaml data.png_root=/scratch-shared/$USER/cifar10_png
python scripts/bench_reads.py /scratch-shared/$USER/cifar10_png/train --processes 1,4,16
```

**Fix.**
- **Few large files** instead of many small ones (one array, or shards of ~100 MB).
- **Copy the dataset to node-local disk** at the start of the job, in one sequential
  copy. Check what's available: on Snellius, `$TMPDIR` on the GPU nodes was not local
  disk; local NVMe needs `--constraint=scratch-node`.
- **Request enough `--mem` for the page cache.** The kernel caches files you read in RAM,
  and on Snellius that cache counts against your job's memory limit. Too little memory,
  and every epoch reads from the network again.

**In continuous-flow-asr.**
- 102 GiB of features under `--mem=64G` ran at 219 samples/s, and 657 once the cache
  could hold them: a 3× longer walltime from asking for too little RAM.
- A cold read was 40 samples/s, against 1,593 warm.
- On GPFS, throughput grew with reads in flight (46 → 215 samples/s from 16 to 128
  processes), then flattened near 0.8 GB/s.
- Copying to node-local NVMe: 765 samples/s.

### Card 11: what you are billed for

**Symptom.** Nothing in your run. It shows up in your budget.

**Fix.**
- **Read your partition's billing weights:**
  `scontrol show partition gpu_mig | grep TRESBillingWeights`. A `gpu_a100` job is
  billed for its *largest* share of a node: 1 GPU, or 18 cores, or 120 GiB of memory.
  Ask for 160 GB of RAM with one GPU, and you pay for two GPUs.
- **Fill what you pay for.** Two runs in that 2-GPU job cost the same as one. A job that
  can't fill an A100 belongs on a MIG slice.
- **A generous walltime is free, a run cut off is not.** Billing is on elapsed time.
- **Screen, then confirm.** Rank options with short, single-seed runs (`StopAtStep`
  keeps the LR schedule of the full run). Run only the winners in full.

**In continuous-flow-asr.**
- With one run per 2-GPU-billed job and the GPU ~37% busy, a 300-epoch run cost ~1,730
  SBU. With the fixes from cards 3, 5 and 6 and two runs per job: ~290.
- Ten runs killed at a 6 h walltime, before finishing, cost 15.4k SBU for nothing.

---

## Two jobs, one GPU

When one job can't keep the GPU busy, a second one can fill the gaps: while job A waits
for its loader or launches its next kernel, job B's kernels run. This is the quickest
win on a laptop, and it needs no change to your code:

```bash
python scripts/run_parallel.py --parallel 2 --grid seed=0,1,2,3
python scripts/run_parallel.py --parallel 1 --grid seed=0,1,2,3     # to compare
```

`run_parallel.py` runs every combination of the `--grid` values, at most `--parallel`
at a time, with one log file and one checkpoint directory per job. It splits the CPU
cores between the jobs' loader workers. At the end it prints the total wall time, which
is the number to compare.

**When it helps.** Card 4 runs (small model, small batch), runs limited by their loader
(card 1) on a machine with spare cores, and sweeps of many short runs.

**When it stops helping.**
- **CPU:** two jobs with 4 workers each on a 6-core laptop are slower than one. The
  script splits the cores for you.
- **RAM:** each job has its own workers, batches and page cache.
- **GPU memory:** both models and both batches have to fit.
- If a job alone is already compute-bound (replay ≈ real and fast), a second one only
  halves the speed of each.

> In continuous-flow-asr, on a laptop RTX 3060: one job kept the GPU 30% busy, two jobs
> 59%. A third added nothing: the machine was out of RAM, and the page cache shrank.

**Long sweeps.** Launch them so that closing the terminal does not kill them:
`setsid nohup python scripts/run_parallel.py ... > sweep.log 2>&1 &`. On WSL, also
raise the VM's memory limit in `.wslconfig`: the default is a fraction of your RAM, and
two jobs can exceed it.

On Snellius, the same idea is a MIG slice (a smaller GPU, billed as one), or two runs in
one job when you are billed for two GPUs anyway (card 11).

---

## Checklist

Before a long run, or before submitting a sweep:

1. `--print-config`: is this the model, the data and the profile you meant?
2. `bench.py`: input-bound or compute-bound? Fix that side first.
3. Batch ladder: the smallest batch within ~5% of the best throughput.
4. bf16-mixed on Ampere or newer; compile for long runs; TF32 only if measured.
5. Nothing per step that needs a Python number: metrics on the GPU, logged per epoch.
6. Workers = cores − 1, `prefetch_factor=1`, few eval workers.
7. Validation on a cadence, at a large batch.
8. On a cluster: enough `--mem` for the data's page cache, and an idea of whether the
   data fits in RAM at all.
9. Read the billing weights, fill the shares you pay for, and give the walltime ~1.5×
   your estimate.
10. A short smoke run that reaches validation, checkpointing and the test, before the
    long one.

---

## Measurement sheet

Fill this in on the hardware you have. Every number above came from a different model
on different data; these are the ones for this repo. Each cell is samples/s from the
`real` row of `bench.py`, except card 4 (best batch from the ladder) and the parallel
row (total wall time).

| Card | Command | laptop | gpu_mig | gpu_a100 |
|---|---|---|---|---|
| baseline | `bench.py` | | | |
| 1 | `bench.py --overlay configs/failures/01_cpu_bound_loader.yaml` | | | |
| 1 fixed | `bench.py data.pipeline=gpu` | | | |
| 2 | `bench.py --overlay configs/failures/02_pageable_copy.yaml` | | | |
| 2 fixed | `... 02_pageable_copy.yaml data.pin_memory=true data.gpu_prefetch=true` | | | |
| 3 | `bench.py --overlay configs/failures/03_host_syncs.yaml` | | | |
| 4 | `bench.py --overlay configs/failures/04_launch_bound.yaml --ladder 32,128,512,2048` | | | |
| 5 | `bench.py --overlay configs/failures/05_full_precision.yaml` | | | |
| 6 | `train.py --overlay configs/failures/06_compile_recompiles.yaml` (count the recompiles) | | | |
| 10 | `bench.py --overlay configs/failures/10_png_files.yaml` | | | |
| 2 jobs | `scripts/run_parallel.py --parallel {1,2} --grid seed=0,1 -- train.max_epochs=2` | | | |
