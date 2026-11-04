"""CIFAR-10 as a LightningDataModule: a train/val split of the training set, and the test set."""

import lightning as L
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
N_TRAIN = 50_000


def cifar_transforms(augment: bool) -> transforms.Compose:
    """Random crop and flip when augmenting, then normalise."""
    steps = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    return transforms.Compose(
        [*(steps if augment else []), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )


class CIFAR10DataModule(L.LightningDataModule):
    """Train/val split of the CIFAR-10 training set, and the test set."""

    def __init__(self, cfg: DictConfig, seed: int = 0) -> None:
        """Keep the `data` section of the config and the split seed."""
        super().__init__()
        self.cfg = cfg
        self.seed = seed

    @property
    def eval_batch_size(self) -> int:
        """Batch size for validation and test."""
        return self.cfg.eval_batch_size or self.cfg.batch_size

    def prepare_data(self) -> None:
        """Download CIFAR-10 once (Lightning calls this on one process only)."""
        CIFAR10(self.cfg.root, train=True, download=True)
        CIFAR10(self.cfg.root, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        """Split the training set into train and val with a fixed seed, and load the test set."""
        perm = torch.randperm(N_TRAIN, generator=torch.Generator().manual_seed(self.seed))
        val_idx, train_idx = perm[: self.cfg.val_size], perm[self.cfg.val_size :]

        train_tf, eval_tf = cifar_transforms(self.cfg.augment), cifar_transforms(False)
        # Two views of the same files: train with augmentation, val without.
        self.train_set = Subset(
            CIFAR10(self.cfg.root, train=True, transform=train_tf), train_idx.tolist()
        )
        self.val_set = Subset(
            CIFAR10(self.cfg.root, train=True, transform=eval_tf), val_idx.tolist()
        )
        self.test_set = CIFAR10(self.cfg.root, train=False, transform=eval_tf)

    def _loader(
        self, dataset: Dataset, batch_size: int, num_workers: int, **kwargs: object
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=num_workers > 0,
            **kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        """Shuffled training batches."""
        return self._loader(
            self.train_set,
            self.cfg.batch_size,
            self.cfg.num_workers,
            shuffle=True,
            drop_last=self.cfg.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation batches, in a fixed order, at the evaluation batch size."""
        return self._loader(self.val_set, self.eval_batch_size, self.cfg.eval_num_workers)

    def test_dataloader(self) -> DataLoader:
        """Test batches, in a fixed order, at the evaluation batch size."""
        return self._loader(self.test_set, self.eval_batch_size, self.cfg.eval_num_workers)
