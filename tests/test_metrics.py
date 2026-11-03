import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassCalibrationError

from dl1.data import GpuBatches
from dl1.metrics import ExpectedCalibrationError, PerClassAccuracy

N_CLASSES = 10


def uneven_batches(seed: int = 0) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Batches of different sizes, as a real loader's last batch would be."""
    g = torch.Generator().manual_seed(seed)
    return [
        (torch.randn(n, N_CLASSES, generator=g) * 3, torch.randint(N_CLASSES, (n,), generator=g))
        for n in (64, 64, 17, 3)
    ]


def test_ece_matches_torchmetrics():
    ours = ExpectedCalibrationError(num_bins=15)
    reference = MulticlassCalibrationError(num_classes=N_CLASSES, n_bins=15, norm="l1")
    for logits, target in uneven_batches():
        ours.update(logits, target)
        reference.update(logits, target)
    torch.testing.assert_close(ours.compute(), reference.compute())


def test_per_class_accuracy_matches_torchmetrics():
    ours = PerClassAccuracy(N_CLASSES)
    reference = MulticlassAccuracy(num_classes=N_CLASSES, average=None)
    for logits, target in uneven_batches():
        ours.update(logits, target)
        reference.update(logits, target)
    torch.testing.assert_close(ours.compute(), reference.compute())


def test_mean_of_batch_accuracies_is_not_the_accuracy():
    # Why metrics keep sums and counts: averaging per-batch results weights a 3-sample
    # batch like a 64-sample one.
    batches = uneven_batches()
    metric = MulticlassAccuracy(num_classes=N_CLASSES, average="micro")
    per_batch = []
    for logits, target in batches:
        metric.update(logits, target)
        per_batch.append((logits.argmax(-1) == target).float().mean())
    assert not torch.isclose(torch.stack(per_batch).mean(), metric.compute())


def test_gpu_batches_shards_are_disjoint_and_equal_in_training():
    n = 103
    images = torch.zeros(n, 3, 32, 32, dtype=torch.uint8)
    labels = torch.arange(n)
    seen = []
    for rank in range(2):
        loader = GpuBatches(images, labels, 10, train=True, augment=True, rank=rank, world_size=2)
        batches = list(loader)
        assert len(batches) == len(loader) == 5  # 51 samples per rank, short batch dropped
        assert all(x.shape == (10, 3, 32, 32) and x.dtype == torch.float32 for x, _ in batches)
        seen.append(torch.cat([y for _, y in batches]))
    assert not set(seen[0].tolist()) & set(seen[1].tolist())


def test_gpu_batches_evaluation_covers_every_sample_once():
    n = 103
    images = torch.zeros(n, 3, 32, 32, dtype=torch.uint8)
    labels = torch.arange(n)
    seen = [
        torch.cat(
            [
                y
                for _, y in GpuBatches(
                    images, labels, 10, train=False, augment=False, rank=rank, world_size=2
                )
            ]
        )
        for rank in range(2)
    ]
    assert sorted(torch.cat(seen).tolist()) == list(range(n))
