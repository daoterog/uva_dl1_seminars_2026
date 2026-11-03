"""How fast can this filesystem hand you files? Reads with N processes x T threads.

    python scripts/bench_reads.py data/cifar10_png/train --processes 1,4,16 --threads 1,4

Each configuration reads the same random sample of files twice. The first pass is
"cold" only if nothing read those files recently, so run it on a fresh set (`--seed`)
each time you want a cold number; the second pass shows what the page cache gives you.

How to read it:
- Throughput that keeps rising with more reads in flight means each read is waiting on
  latency (typical of network filesystems such as GPFS): more workers help.
- Throughput flat from some point on means you have hit the filesystem's bandwidth:
  more workers do not help; fewer, bigger files or local disk do.
- A second pass much faster than the first means the page cache holds your data. On a
  cluster, that cache counts against your job's --mem: too little memory and every epoch
  is a cold pass.
"""

import argparse
import random
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path


def read_all(paths: list[str], threads: int) -> int:
    """Read every file in `paths` with `threads` threads; return the bytes read."""

    def read(path: str) -> int:
        with open(path, "rb") as f:
            return len(f.read())

    with ThreadPoolExecutor(threads) as pool:
        return sum(pool.map(read, paths))


def one_pass(paths: list[str], processes: int, threads: int) -> tuple[float, int]:
    """Split `paths` over `processes` processes; return (seconds, bytes)."""
    chunks = [paths[i::processes] for i in range(processes)]
    started = time.perf_counter()
    with ProcessPoolExecutor(processes) as pool:
        total = sum(pool.map(read_all, chunks, [threads] * processes))
    return time.perf_counter() - started, total


def main() -> None:
    """Time a cold and a warm pass for each processes x threads configuration."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("directory", help="directory of files to read")
    parser.add_argument("--n-files", type=int, default=5000, help="files per configuration")
    parser.add_argument("--processes", default="1,4,16", help="comma-separated process counts")
    parser.add_argument("--threads", default="1,4", help="comma-separated threads per process")
    parser.add_argument("--seed", type=int, default=0, help="change it for files not read before")
    args = parser.parse_args()

    files = sorted(str(p) for p in Path(args.directory).iterdir() if p.is_file())
    rng = random.Random(args.seed)
    print(f"{'procs':>6}{'threads':>8}{'cold files/s':>14}{'warm files/s':>14}{'cold MB/s':>11}")
    for processes in map(int, args.processes.split(",")):
        for threads in map(int, args.threads.split(",")):
            # A fresh sample per configuration, so no configuration reads the files an
            # earlier one already pulled into the page cache.
            sample = rng.sample(files, min(args.n_files, len(files)))
            cold, size = one_pass(sample, processes, threads)
            warm, _ = one_pass(sample, processes, threads)
            n, mb_per_s = len(sample), size / cold / 1e6
            print(f"{processes:>6}{threads:>8}{n / cold:>14.0f}{n / warm:>14.0f}{mb_per_s:>11.1f}")


if __name__ == "__main__":
    main()
