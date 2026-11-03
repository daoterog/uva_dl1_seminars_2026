"""Run several training jobs on one machine, at most `--parallel` at a time.

    python scripts/run_parallel.py --grid optim.lr=0.05,0.1,0.2 --grid seed=0,1
    python scripts/run_parallel.py --parallel 2 --grid seed=0,1,2,3 -- train.max_epochs=10
    python scripts/run_parallel.py --dry-run --grid model.name=mlp,resnet18

Every combination of the `--grid` values is one job; anything after `--` goes to every
job. Each job writes its output to logs/<group>/<job>.log and its checkpoints to its own
directory, and all of them share one wandb group.

Why run two at once: a small job on a laptop often cannot keep the GPU busy on its own.
It waits on its data loader, or each step is too little work. A second job fills those
gaps. In continuous-flow-asr, one job left a laptop RTX 3060 about 30% busy and two
reached 59%. A third added nothing: by then the machine was out of RAM, not GPU. Start
at 2, and compare the total wall time this prints against `--parallel 1` on the same
grid, which is the number that matters.

The jobs share the CPU too, so each gets an equal share of the data-loading workers
(cores - 1, split between the jobs) unless you set `data.num_workers` yourself.

Long sweeps: launch them so that closing the terminal does not kill them,

    setsid nohup python scripts/run_parallel.py --grid seed=0,1,2,3 > sweep.log 2>&1 &

and watch `free -g` (RAM) and `nvidia-smi` (GPU memory) while the first pair starts.
"""

import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_grid(specs: list[str]) -> list[list[str]]:
    """`["a=1,2", "b=x"]` -> `[["a=1", "b=x"], ["a=2", "b=x"]]`, one list per job."""
    axes = []
    for spec in specs:
        key, _, values = spec.partition("=")
        axes.append([f"{key}={v}" for v in values.split(",")])
    return [list(combo) for combo in itertools.product(*axes)]


def job_name(assignment: list[str]) -> str:
    """A filesystem-safe name from a job's overrides, e.g. `optim.lr-0.1_seed-0`."""
    return "_".join(a.replace("=", "-").replace("/", "-") for a in assignment) or "job"


def main() -> None:
    """Build the job list, then keep up to `--parallel` of them running until all finish."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--parallel", type=int, default=2, help="jobs running at once")
    parser.add_argument("--grid", action="append", default=[], help="key=v1,v2,... (repeatable)")
    parser.add_argument("--group", default=time.strftime("parallel-%Y%m%d-%H%M%S"))
    parser.add_argument("--dry-run", action="store_true", help="print the commands only")
    parser.add_argument("common", nargs="*", help="overrides for every job (after --)")
    args = parser.parse_args()

    workers = max(1, ((os.cpu_count() or 2) - 1) // args.parallel)
    shared = list(args.common)
    if not any(a.startswith("data.num_workers=") for a in shared):
        shared.append(f"data.num_workers={workers}")
    log_dir = ROOT / "logs" / args.group
    jobs = []
    for assignment in parse_grid(args.grid):
        name = job_name(assignment)
        command = [
            sys.executable, str(ROOT / "train.py"), *shared, *assignment,
            f"logging.group={args.group}",
            f"logging.checkpoint_dir=checkpoints/{args.group}/{name}",
        ]  # fmt: skip
        jobs.append((name, command))

    if args.dry_run:
        for _, command in jobs:
            print(" ".join(command))
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} jobs, {args.parallel} at a time, {workers} loader workers each; "
          f"logs in {log_dir}")  # fmt: skip
    pending, running = list(jobs), {}
    started = time.perf_counter()
    try:
        while pending or running:
            while pending and len(running) < args.parallel:
                name, command = pending.pop(0)
                log = open(log_dir / f"{name}.log", "w")  # noqa: SIM115 - closed when the job ends
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
                running[name] = (process, log, time.perf_counter())
                print(f"  started  {name}")
            for name, (process, log, t0) in list(running.items()):
                if process.poll() is not None:
                    log.close()
                    status = "ok" if process.returncode == 0 else f"FAILED ({process.returncode})"
                    print(f"  finished {name}: {status} after {time.perf_counter() - t0:.0f}s")
                    del running[name]
            time.sleep(1)
    except KeyboardInterrupt:
        for process, log, _ in running.values():
            process.terminate()
            log.close()
        raise
    print(f"all {len(jobs)} jobs done in {time.perf_counter() - started:.0f}s wall time")


if __name__ == "__main__":
    main()
