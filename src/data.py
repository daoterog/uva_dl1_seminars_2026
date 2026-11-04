"""CIFAR-10 as a LightningDataModule: the data cells of notebooks/prototyping.ipynb, in hooks."""

from pathlib import Path

import lightning as L
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def cifar_transforms(augment: bool) -> transforms.Compose:
    """Random crop and flip when augmenting, then normalise."""
    steps = []
    if augment:
        steps += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)


class CIFAR10DataModule(L.LightningDataModule):
    """A train/val split of the CIFAR-10 training set, and the test set."""

    def __init__(self, cfg: DictConfig, seed: int = 0) -> None:
        """Keep the `data` section of the config, and the seed of the train/val split."""
        super().__init__()
        self.cfg = cfg
        self.seed = seed

    def prepare_data(self) -> None:
        """Download CIFAR-10. Lightning calls this once, on a single process."""
        Path(self.cfg.root).mkdir(parents=True, exist_ok=True)
        CIFAR10(self.cfg.root, train=True, download=True)
        CIFAR10(self.cfg.root, train=False, download=True)

    def setup(self, stage: str) -> None:
        """Build the datasets and the split. Lightning calls this on every process."""
        # ##########################################################################################
        # NOTE: evaluation never augments, unless it is deliberate.
        #
        # Validation and test must be deterministic: the same model has to get the same score
        # every time, or comparing runs and picking the best checkpoint mean nothing. If you do
        # want a stochastic evaluation (test-time augmentation, say), make it intentional and
        # controlled: its own setting, a fixed seed, and reported as such.
        # ##########################################################################################
        train_tf, eval_tf = cifar_transforms(self.cfg.augment), cifar_transforms(augment=False)
        train_full = CIFAR10(self.cfg.root, train=True, transform=train_tf)
        val_full = CIFAR10(self.cfg.root, train=True, transform=eval_tf)
        self.test_set = CIFAR10(self.cfg.root, train=False, transform=eval_tf)

        # A fixed split, so every run validates on the same images.
        generator = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(len(train_full), generator=generator)
        self.train_set = Subset(train_full, perm[self.cfg.val_size :].tolist())
        self.val_set = Subset(val_full, perm[: self.cfg.val_size].tolist())

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        """Batches of `batch_size`, loaded by `num_workers` worker processes."""
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.cfg.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        """Shuffled training batches."""
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Validation batches, in a fixed order."""
        return self._loader(self.val_set, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Test batches, in a fixed order."""
        return self._loader(self.test_set, shuffle=False)
