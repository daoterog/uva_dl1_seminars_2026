"""A custom torchmetrics metric.

A torchmetrics `Metric` keeps *sufficient statistics* as state (sums and counts), not
per-batch results. That is what makes it correct in two places a hand-written average is
not:

- **Batches of different sizes.** Averaging per-batch accuracies weights a short last
  batch like a full one. Summing correct predictions and counts, then dividing once at
  the end, does not.
- **Several processes (DDP).** Each process sees a different shard of the data. Every
  state declared with `add_state(..., dist_reduce_fx="sum")` is summed across processes
  when `compute()` runs, so the result is the metric over the whole dataset, as if one
  process had seen it all. A mean of per-process accuracies would be wrong in the same
  way as a mean of per-batch ones.

The metric here updates with on-device tensor ops only: no `.item()`, no `.cpu()`, no
Python loop over the batch, so updating it every step never makes the CPU wait for the GPU.
"""

import torch
from torchmetrics import Metric


class PerClassAccuracy(Metric):
    """Accuracy of each class separately, as a (num_classes,) tensor.

    Kept as correct and total counts per class, not as per-class accuracies: a class that
    is rare in one process's shard would otherwise pull the merged value as hard as a
    common one.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, num_classes: int, **kwargs: object) -> None:
        """Set up correct and total counts for `num_classes` classes."""
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.add_state("correct", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """Add a batch of (B, C) logits and (B,) integer targets."""
        hit = (logits.argmax(dim=-1) == target).float()
        self.correct.scatter_add_(0, target, hit)
        self.total.scatter_add_(0, target, torch.ones_like(hit))

    def compute(self) -> torch.Tensor:
        """Correct over total per class; NaN for a class that never appeared."""
        return self.correct / self.total.where(self.total > 0, torch.nan)
