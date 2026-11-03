"""Overlapping the host-to-device copy of the next batch with the current step."""

from collections.abc import Iterable, Iterator

import torch


class CudaPrefetcher:
    """A loader whose batches arrive already on the GPU, copied while the previous one trains.

    Without it, the batch is copied to the GPU on the compute stream right before the
    step, and the step cannot start until the copy ends. Here the next batch's copy is
    issued on a side stream as soon as the current batch is handed out, so it overlaps
    with that batch's forward and backward. Lightning then finds the batch already on the
    device and does nothing.

    Needs `pin_memory=True` on the wrapped loader: a non-blocking copy from pageable
    memory is a synchronous copy, and nothing is gained.

    Single process only: Lightning cannot see the DataLoader inside to give it a
    DistributedSampler.
    """

    def __init__(self, loader: Iterable, device: torch.device) -> None:
        """Wrap `loader`; its length, workers and sampler are unchanged."""
        self.loader = loader
        self.device = device

    def __len__(self) -> int:
        """Batches per epoch, as the wrapped loader reports them."""
        return len(self.loader)  # type: ignore[arg-type]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        """Yield each batch on the GPU, with the next one already in flight."""
        stream = torch.cuda.Stream(self.device)
        batches = iter(self.loader)

        def fetch() -> tuple[torch.Tensor, ...] | None:
            batch = next(batches, None)
            if batch is None:
                return None
            with torch.cuda.stream(stream):
                return tuple(t.to(self.device, non_blocking=True) for t in batch)

        upcoming = fetch()
        while upcoming is not None:
            compute = torch.cuda.current_stream(self.device)
            compute.wait_stream(stream)
            batch = upcoming
            for t in batch:
                # Allocated on the side stream, used on the compute stream: without this
                # the caching allocator could hand its memory out again while in use.
                t.record_stream(compute)
            upcoming = fetch()
            yield batch
