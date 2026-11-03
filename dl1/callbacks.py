"""Callbacks that make a run's efficiency visible, and keep a cluster run honest."""

import time

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback


class DataWaitMonitor(Callback):
    """Per epoch: the share of wall time spent waiting for the next batch, and samples/s.

    The wait is the time from the end of one training step to the start of the next,
    which is where Lightning fetches the next batch and moves it to the device. Near 0
    means the input keeps up; a large share means the GPU is waiting for the input, and no
    change to the model will make the run faster.

    This is measured in the training process, so it works where `nvidia-smi` cannot help
    (on a MIG slice it does not report utilisation at all). It reads the host's clock, and
    the GPU runs asynchronously: treat it as "how long the loop stood still for input",
    and confirm a diagnosis with bench.py, which synchronises.
    """

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Start the epoch's clock."""
        self.epoch_start = self.last_end = time.perf_counter()
        self.waited = 0.0
        self.samples = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: tuple[torch.Tensor, ...],
        batch_idx: int,
    ) -> None:
        """Count the time since the previous step ended as waiting for this batch."""
        self.waited += time.perf_counter() - self.last_end
        self.samples += len(batch[0])

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: object,
        batch: tuple[torch.Tensor, ...],
        batch_idx: int,
    ) -> None:
        """Mark the end of the step."""
        self.last_end = time.perf_counter()

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log the epoch's wait share, throughput and peak GPU memory."""
        wall = time.perf_counter() - self.epoch_start
        stats = {
            "perf/data_wait_frac": self.waited / wall,
            "perf/samples_per_s": self.samples / wall,
        }
        if torch.cuda.is_available():
            stats["perf/peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
        pl_module.log_dict(stats)


class Heartbeat(Callback):
    """Print one plain line every `every` steps, for SLURM logs without a progress bar.

    It prints the step and the wall clock only, never a metric: reading a metric's value
    would make the CPU wait for the GPU. If this line keeps coming and wandb stops, the
    logger died, not the training.
    """

    def __init__(self, every: int = 500) -> None:
        """Print every `every` optimiser steps."""
        self.every = every
        self.start = time.perf_counter()

    def on_train_batch_end(self, trainer: L.Trainer, *_args: object) -> None:
        """Print a line on every `every`-th step."""
        step = trainer.global_step
        if step % self.every == 0 and trainer.is_global_zero:
            elapsed = time.perf_counter() - self.start
            print(
                f"[heartbeat] step {step} epoch {trainer.current_epoch} {elapsed:.0f}s", flush=True
            )


class StopAtStep(Callback):
    """End training at a step, leaving the LR schedule sized for the full run.

    `Trainer(max_steps=...)` would shrink the one-cycle schedule to fit. Stopping here
    instead gives a cheap run that matches the full one step for step up to where it
    stops, which is what you want when screening options before paying for full runs.
    """

    def __init__(self, step: int) -> None:
        """Stop once `step` optimiser steps have run."""
        self.step = step

    def on_train_batch_end(self, trainer: L.Trainer, *_args: object) -> None:
        """Ask the trainer to stop once the step is reached."""
        if trainer.global_step >= self.step:
            trainer.should_stop = True
