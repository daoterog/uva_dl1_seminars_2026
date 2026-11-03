"""Custom torchmetrics metrics.

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

Both metrics here update with on-device tensor ops only: no `.item()`, no `.cpu()`, no
Python loop over the batch, so updating them every step never makes the CPU wait for
the GPU.
"""

import torch
from torchmetrics import Metric


class ExpectedCalibrationError(Metric):
    """How far the model's confidence is from its accuracy, averaged over confidence bins.

    Predictions are binned by their top-class probability. In each bin, compare the mean
    confidence with the fraction that were correct; ECE is the sample-weighted mean of
    the absolute gaps. 0 is perfectly calibrated.

    The state is three per-bin sums, which is all ECE needs, and sums reduce exactly
    across processes.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, num_bins: int = 15, **kwargs: object) -> None:
        """Set up `num_bins` equal-width confidence bins over [0, 1]."""
        super().__init__(**kwargs)
        self.num_bins = num_bins
        self.register_buffer("boundaries", torch.linspace(0, 1, num_bins + 1), persistent=False)
        self.add_state("confidence_sum", default=torch.zeros(num_bins), dist_reduce_fx="sum")
        self.add_state("correct_sum", default=torch.zeros(num_bins), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(num_bins), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """Add a batch of (B, C) logits and (B,) integer targets."""
        confidence, prediction = logits.float().softmax(dim=-1).max(dim=-1)
        correct = (prediction == target).float()
        # Bin k holds [k/n, (k+1)/n); a confidence of exactly 1 goes in the top bin.
        bins = (torch.bucketize(confidence, self.boundaries, right=True) - 1).clamp(
            0, self.num_bins - 1
        )
        self.confidence_sum.scatter_add_(0, bins, confidence)
        self.correct_sum.scatter_add_(0, bins, correct)
        self.count.scatter_add_(0, bins, torch.ones_like(confidence))

    def compute(self) -> torch.Tensor:
        """Sum over bins of |confidence - accuracy|, weighted by each bin's share of samples."""
        return (self.confidence_sum - self.correct_sum).abs().sum() / self.count.sum()


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
