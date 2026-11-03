"""The metrics under real DDP: two CPU processes over gloo, each seeing a different shard.

This is what `dist_reduce_fx="sum"` is for. Each process updates its metric with its own
shard; `compute()` sums the states across processes first, so both processes get the
metric over the whole dataset. The shards are deliberately uneven (a DistributedSampler
without padding, or a last batch, would make them so).
"""

from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dl1.metrics import ExpectedCalibrationError, PerClassAccuracy

N_CLASSES = 10
SHARDS = (37, 63)


def dataset() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    n = sum(SHARDS)
    return torch.randn(n, N_CLASSES, generator=g) * 3, torch.randint(N_CLASSES, (n,), generator=g)


def metrics() -> dict:
    return {"ece": ExpectedCalibrationError(), "per_class": PerClassAccuracy(N_CLASSES)}


def worker(rank: int, tmp: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{tmp}/rendezvous", rank=rank, world_size=2)
    logits, target = dataset()
    start = sum(SHARDS[:rank])
    shard = slice(start, start + SHARDS[rank])
    results = {}
    for name, metric in metrics().items():
        metric.update(logits[shard], target[shard])
        results[name] = metric.compute()  # syncs: every process gets the global value
    own_acc = (logits[shard].argmax(-1) == target[shard]).float().mean()
    results["own_acc"] = own_acc
    torch.save(results, Path(tmp) / f"rank{rank}.pt")
    dist.destroy_process_group()


def test_synced_metrics_equal_single_process(tmp_path: Path):
    mp.spawn(worker, args=(str(tmp_path),), nprocs=2, join=True)
    per_rank = [torch.load(tmp_path / f"rank{r}.pt") for r in range(2)]

    logits, target = dataset()
    for name, metric in metrics().items():
        metric.update(logits, target)
        expected = metric.compute()
        for result in per_rank:
            torch.testing.assert_close(result[name], expected, equal_nan=True)

    # What a hand-rolled "average the per-process accuracies" would report instead.
    naive = torch.stack([r["own_acc"] for r in per_rank]).mean()
    overall = (logits.argmax(-1) == target).float().mean()
    assert not torch.isclose(naive, overall)
