"""Time the training step three ways, to tell an input-bound run from a compute-bound one.

    python bench.py                                   # the three rows at the profile's batch
    python bench.py --overlay configs/failures/01_cpu_bound_loader.yaml
    python bench.py --rows real --ladder 64,128,256,512
    python bench.py data.pipeline=gpu train.compile=true

Rows:
- **real**: the actual training loop (Lightning, your data pipeline, your model).
- **replay**: the same loop fed one batch already on the GPU, over and over. No loader,
  no copy: the fastest this model can train at this batch size on this GPU.
- **loader**: the data pipeline alone, each batch moved to the GPU, no model.

How to read it: if replay is much faster than real, the GPU is waiting for its input,
and the fix is in the data pipeline (cookbook cards 1, 2, 8, 10). If replay is about
as fast as real, the GPU itself is the limit, and the fix is in the step (cards 3-6).

Each row runs in a fresh process, so no row inherits another's memory, compiled graphs
or warm caches. Timing starts after `--warmup` steps (compilation, worker start-up and the
first allocations are not the steady state) and both ends synchronise with the GPU.
"""

import argparse
import json
import subprocess
import sys
import time

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback

from dl1.config import build_config
from dl1.data import CIFAR10DataModule
from dl1.module import Classifier

MARK = "BENCH "


def sync() -> None:
    """Wait for the GPU, if there is one: the clock must not stop before the work does."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class StepTimer(Callback):
    """Times `steps` steps after `warmup`, and the share of them spent waiting for input."""

    def __init__(self, warmup: int, steps: int) -> None:
        """Count `warmup` untimed steps, then time `steps`."""
        self.warmup, self.steps = warmup, steps
        self.done = 0
        self.started: float | None = None
        self.ended: float | None = None
        self.waited = 0.0
        self.last_end = 0.0

    def on_train_batch_start(self, *_args: object) -> None:
        """Count the gap since the last step as waiting for input."""
        if self.started is not None:
            self.waited += time.perf_counter() - self.last_end

    def on_train_batch_end(self, *_args: object) -> None:
        """Start the clock after warmup; stop it after the last timed step."""
        self.done += 1
        if self.done == self.warmup:
            sync()
            self.started = time.perf_counter()
        elif self.done == self.warmup + self.steps:
            sync()
            self.ended = time.perf_counter()
        self.last_end = time.perf_counter()


class ReplayLoader:
    """The same on-device batch, `steps` times, sized like a DataLoader for Lightning."""

    def __init__(self, batch: tuple[torch.Tensor, ...], steps: int) -> None:
        """Hold the batch."""
        self.batch, self.steps = batch, steps

    def __len__(self) -> int:
        """Batches per epoch."""
        return self.steps

    def __iter__(self):  # noqa: ANN204 - a plain iterator of batches
        """The batch, `steps` times."""
        return iter([self.batch] * self.steps)


def first_batch(dm: CIFAR10DataModule, device: torch.device) -> tuple[torch.Tensor, ...]:
    """One training batch from the real pipeline, moved to `device`."""
    batch = next(iter(dm.train_dataloader()))
    return tuple(t.to(device) for t in batch)


def time_fit(cfg: object, args: argparse.Namespace, replay: bool) -> dict:
    """Time the training loop, on the real pipeline or on one replayed batch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = CIFAR10DataModule(cfg.data, seed=cfg.seed, device=device)
    timer = StepTimer(args.warmup, args.steps)
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        precision=cfg.train.precision,
        max_steps=args.warmup + args.steps,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        benchmark=cfg.train.cudnn_benchmark,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[timer],
    )
    model = Classifier(cfg)
    if replay:
        dm.prepare_data()
        dm.setup("fit")
        loader = ReplayLoader(first_batch(dm, device), args.warmup + args.steps)
        trainer.fit(model, train_dataloaders=loader)
    else:
        trainer.fit(model, datamodule=dm)
    seconds = timer.ended - timer.started
    return {"seconds": seconds, "steps": args.steps, "waited": timer.waited}


def time_loader(cfg: object, args: argparse.Namespace) -> dict:
    """Time the data pipeline alone: fetch each batch and move it to the device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = CIFAR10DataModule(cfg.data, seed=cfg.seed, device=device)
    dm.prepare_data()
    dm.setup("fit")
    done, started = 0, None
    while started is None or done < args.warmup + args.steps:
        for batch in dm.train_dataloader():
            [t.to(device, non_blocking=True) for t in batch]
            done += 1
            if done == args.warmup:
                sync()
                started = time.perf_counter()
            if done == args.warmup + args.steps:
                break
    sync()
    return {"seconds": time.perf_counter() - started, "steps": args.steps, "waited": None}


def child(args: argparse.Namespace, overrides: list[str]) -> None:
    """Run one row in this process and print its result as one JSON line."""
    cfg = build_config(args.where, args.overlay, [*overrides, "logging.wandb_mode=disabled"])
    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision(cfg.train.matmul_precision)
    if args.child == "loader":
        result = time_loader(cfg, args)
    else:
        result = time_fit(cfg, args, replay=args.child == "replay")
    result["batch_size"] = cfg.data.batch_size
    if torch.cuda.is_available():
        result["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
    print(MARK + json.dumps(result), flush=True)


def run_row(
    row: str, batch_size: int | None, args: argparse.Namespace, overrides: list[str]
) -> dict:
    """Run one row in a fresh process and collect its result."""
    extra = [] if batch_size is None else [f"data.batch_size={batch_size}"]
    command = [
        sys.executable,
        __file__,
        "--child",
        row,
        "--warmup",
        str(args.warmup),
        "--steps",
        str(args.steps),
        *(["--where", args.where] if args.where else []),
        *[f"--overlay={o}" for o in args.overlay],
        *overrides,
        *extra,
    ]
    out = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    for line in out.stdout.splitlines():
        if line.startswith(MARK):
            return {"row": row, **json.loads(line[len(MARK) :])}
    return {"row": row, "batch_size": batch_size, "failed": out.returncode}


def print_table(results: list[dict]) -> None:
    """One line per row: ms/step, samples/s, input wait, peak GPU memory."""
    print(f"\n{'row':<8}{'batch':>7}{'ms/step':>10}{'samples/s':>11}{'wait':>7}{'peak GiB':>10}")
    for r in results:
        if "failed" in r:
            print(f"{r['row']:<8}{r['batch_size'] or '':>7}   failed (exit {r['failed']}; OOM?)")
            continue
        ms = 1000 * r["seconds"] / r["steps"]
        rate = r["steps"] * r["batch_size"] / r["seconds"]
        wait = "-" if r["waited"] is None else f"{r['waited'] / r['seconds']:.0%}"
        peak = f"{r['peak_gib']:.2f}" if "peak_gib" in r else "-"
        print(f"{r['row']:<8}{r['batch_size']:>7}{ms:>10.1f}{rate:>11.0f}{wait:>7}{peak:>10}")


def main() -> None:
    """Parse flags, run every requested row in its own process, print the table."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--where", help="a configs/where/ profile (default: detected)")
    parser.add_argument(
        "--overlay", action="append", default=[], help="extra YAML, merged in order"
    )
    parser.add_argument("--rows", default="real,replay,loader", help="comma-separated rows to run")
    parser.add_argument("--ladder", help="comma-separated batch sizes, each row run at each")
    parser.add_argument("--warmup", type=int, default=30, help="untimed steps first")
    parser.add_argument("--steps", type=int, default=100, help="timed steps")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args, overrides = parser.parse_known_args()
    if args.child:
        child(args, overrides)
        return

    sizes = [int(b) for b in args.ladder.split(",")] if args.ladder else [None]
    results = [run_row(row, b, args, overrides) for b in sizes for row in args.rows.split(",")]
    print_table(results)


if __name__ == "__main__":
    main()
