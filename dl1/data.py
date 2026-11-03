"""CIFAR-10 for Lightning, through one of three input pipelines.

- **classic**: torchvision transforms on PIL images, in DataLoader worker processes. The
  usual way, and CPU-bound once the GPU is fast: every image is decoded, cropped,
  flipped and normalised by a Python call on a CPU core.
- **gpu**: the whole dataset (150 MB as uint8) lives on the GPU, and each batch is
  indexed, augmented and normalised there in a few batched ops. No workers, no
  host-to-device copy. Only possible because the dataset fits in GPU memory; when it does,
  nothing is faster.
- **png**: one PNG file per image, as a dataset of photos would be stored. On a network
  filesystem, opening 50,000 small files is its own bottleneck.

All three draw the same train/validation split from `seed`.
"""

from collections.abc import Iterator
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from dl1.prefetch import CudaPrefetcher

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
N_TRAIN = 50_000


def cifar_transforms(augment: bool) -> transforms.Compose:
    """The classic CPU-side transforms: random crop and flip, then normalise."""
    steps = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    return transforms.Compose(
        [*(steps if augment else []), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )


class PngCifar(Dataset):
    """CIFAR-10 stored as one `<index>_<label>.png` per image (see scripts/export_pngs.py)."""

    def __init__(self, root: str | Path, transform: transforms.Compose) -> None:
        """Index the PNGs under `root`; names sort by image index."""
        self.files = sorted(Path(root).glob("*.png"))
        if len(self.files) != N_TRAIN:
            raise FileNotFoundError(
                f"expected {N_TRAIN} PNGs in {root}, found {len(self.files)}; "
                "run scripts/export_pngs.py first"
            )
        self.labels = [int(f.stem.split("_")[1]) for f in self.files]
        self.transform = transform

    def __len__(self) -> int:
        """Number of images."""
        return len(self.files)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        """Open, decode and transform image `i`."""
        with Image.open(self.files[i]) as img:
            return self.transform(img.convert("RGB")), self.labels[i]


class GpuBatches:
    """Batches drawn from tensors that already live on the GPU, augmented on the GPU.

    Under DDP, each process takes every `world_size`-th sample of the same permutation,
    so the shards are disjoint and, in training, equal in size (an uneven number of
    training steps across processes would hang DDP's gradient all-reduce).
    """

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int,
        *,
        train: bool,
        augment: bool,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        steps_per_epoch: int | None = None,
    ) -> None:
        """Hold (N, 3, 32, 32) uint8 images and (N,) labels, both on the target device."""
        self.images, self.labels = images, labels
        self.batch_size = batch_size
        self.train, self.augment = train, augment
        self.seed, self.rank, self.world_size = seed, rank, world_size
        self.steps_per_epoch = steps_per_epoch
        self.epoch = 0
        device = images.device
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    def _num_samples(self) -> int:
        n = len(self.labels)
        if self.steps_per_epoch is not None:
            return self.steps_per_epoch * self.batch_size
        if self.train:
            return n // self.world_size
        return len(range(self.rank, n, self.world_size))

    def __len__(self) -> int:
        """Batches per epoch on this process; training drops the short last batch."""
        n = self._num_samples()
        return n // self.batch_size if self.train else -(-n // self.batch_size)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """One epoch of (images, labels) batches."""
        n, device = len(self.labels), self.labels.device
        generator = torch.Generator(device=device).manual_seed(self.seed + self.epoch)
        self.epoch += 1
        if self.steps_per_epoch is not None:
            index = torch.randint(n, (self._num_samples(),), generator=generator, device=device)
        elif self.train:
            index = torch.randperm(n, generator=generator, device=device)
            index = index[: n // self.world_size * self.world_size][self.rank :: self.world_size]
        else:
            index = torch.arange(self.rank, n, self.world_size, device=device)
        for b in range(len(self)):
            batch = index[b * self.batch_size : (b + 1) * self.batch_size]
            yield self._prepare(self.images[batch], generator), self.labels[batch]

    def _prepare(self, x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """uint8 -> normalised float, with a random 4-pixel-padded crop and flip in training."""
        x = x.float().div_(255)
        if self.augment:
            b, device = x.shape[0], x.device
            padded = F.pad(x, (4, 4, 4, 4))
            offsets = torch.randint(0, 9, (2, b), generator=generator, device=device)
            grid = torch.arange(32, device=device)
            rows = (offsets[0, :, None] + grid)[:, None, :, None]
            cols = (offsets[1, :, None] + grid)[:, None, None, :]
            x = padded[
                torch.arange(b, device=device)[:, None, None, None],
                torch.arange(3, device=device)[None, :, None, None],
                rows,
                cols,
            ]
            flip = torch.rand(b, generator=generator, device=device) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        return (x - self.mean) / self.std


def to_nchw(data: object) -> torch.Tensor:
    """Torchvision's (N, 32, 32, 3) uint8 array -> a contiguous (N, 3, 32, 32) tensor."""
    return torch.as_tensor(data).permute(0, 3, 1, 2).contiguous()


class CIFAR10DataModule(L.LightningDataModule):
    """Train/val split of the CIFAR-10 training set, and the test set, per `cfg.pipeline`."""

    def __init__(self, cfg: DictConfig, seed: int = 0, device: torch.device | None = None) -> None:
        """Keep the `data` section of the config and the split seed.

        `device` is only for use without a Trainer (bench.py); with one, its device wins.
        """
        super().__init__()
        self.cfg = cfg
        self.seed = seed
        self.fallback_device = device or torch.device("cpu")

    @property
    def eval_batch_size(self) -> int:
        """Batch size for validation and test."""
        return self.cfg.eval_batch_size or self.cfg.batch_size

    def _rank(self) -> tuple[int, int]:
        if self.trainer is None:
            return 0, 1
        return self.trainer.global_rank, self.trainer.world_size

    def _device(self) -> torch.device:
        return self.trainer.strategy.root_device if self.trainer else self.fallback_device

    def prepare_data(self) -> None:
        """Download CIFAR-10 once (Lightning calls this on one process only)."""
        CIFAR10(self.cfg.root, train=True, download=True)
        CIFAR10(self.cfg.root, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        """Build the split, then the datasets (or the on-GPU tensors) for the pipeline."""
        perm = torch.randperm(N_TRAIN, generator=torch.Generator().manual_seed(self.seed))
        self.val_idx = perm[: self.cfg.val_size]
        self.train_idx = perm[self.cfg.val_size :]
        if self.cfg.subset is not None:
            self.train_idx = self.train_idx[: self.cfg.subset]

        if self.cfg.pipeline == "gpu":
            device = self._device()
            train = CIFAR10(self.cfg.root, train=True)
            test = CIFAR10(self.cfg.root, train=False)
            images = to_nchw(train.data).to(device)
            labels = torch.as_tensor(train.targets).to(device)
            self.gpu = {
                "train": (images[self.train_idx.to(device)], labels[self.train_idx.to(device)]),
                "val": (images[self.val_idx.to(device)], labels[self.val_idx.to(device)]),
                "test": (to_nchw(test.data).to(device), torch.as_tensor(test.targets).to(device)),
            }
            return

        train_tf, eval_tf = cifar_transforms(self.cfg.augment), cifar_transforms(False)
        if self.cfg.pipeline == "png":
            source_train = PngCifar(Path(self.cfg.png_root) / "train", train_tf)
            source_val = PngCifar(Path(self.cfg.png_root) / "train", eval_tf)
        elif self.cfg.pipeline == "classic":
            source_train = CIFAR10(self.cfg.root, train=True, transform=train_tf)
            source_val = CIFAR10(self.cfg.root, train=True, transform=eval_tf)
        else:
            raise ValueError(f"unknown data.pipeline {self.cfg.pipeline!r}")
        self.train_set = Subset(source_train, self.train_idx.tolist())
        self.val_set = Subset(source_val, self.val_idx.tolist())
        self.test_set = CIFAR10(self.cfg.root, train=False, transform=eval_tf)

    def _loader(
        self, dataset: Dataset, batch_size: int, num_workers: int, **kwargs: object
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers and num_workers > 0,
            prefetch_factor=self.cfg.prefetch_factor if num_workers > 0 else None,
            **kwargs,
        )

    def _maybe_prefetch(self, loader: DataLoader) -> DataLoader | CudaPrefetcher:
        if not self.cfg.gpu_prefetch or self._device().type != "cuda":
            return loader
        if self._rank()[1] > 1:
            raise ValueError("data.gpu_prefetch is single-GPU only: it hides the sampler from DDP")
        return CudaPrefetcher(loader, self._device())

    def _gpu_batches(self, split: str, batch_size: int, train: bool) -> GpuBatches:
        rank, world_size = self._rank()
        images, labels = self.gpu[split]
        return GpuBatches(
            images,
            labels,
            batch_size,
            train=train,
            augment=train and self.cfg.augment,
            seed=self.seed,
            rank=rank,
            world_size=world_size,
            steps_per_epoch=self.cfg.steps_per_epoch if train else None,
        )

    def train_dataloader(self) -> DataLoader | CudaPrefetcher | GpuBatches:
        """Shuffled training batches; a fixed number per epoch if `steps_per_epoch` is set."""
        if self.cfg.pipeline == "gpu":
            return self._gpu_batches("train", self.cfg.batch_size, train=True)
        sampler = None
        if self.cfg.steps_per_epoch is not None:
            sampler = RandomSampler(
                self.train_set,
                replacement=True,
                num_samples=self.cfg.steps_per_epoch * self.cfg.batch_size,
            )
        loader = self._loader(
            self.train_set,
            self.cfg.batch_size,
            self.cfg.num_workers,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=self.cfg.drop_last,
        )
        return self._maybe_prefetch(loader)

    def val_dataloader(self) -> DataLoader | CudaPrefetcher | GpuBatches:
        """Validation batches, in a fixed order, at the evaluation batch size."""
        if self.cfg.pipeline == "gpu":
            return self._gpu_batches("val", self.eval_batch_size, train=False)
        loader = self._loader(self.val_set, self.eval_batch_size, self.cfg.eval_num_workers)
        return self._maybe_prefetch(loader)

    def test_dataloader(self) -> DataLoader | CudaPrefetcher | GpuBatches:
        """Test batches, in a fixed order, at the evaluation batch size."""
        if self.cfg.pipeline == "gpu":
            return self._gpu_batches("test", self.eval_batch_size, train=False)
        loader = self._loader(self.test_set, self.eval_batch_size, self.cfg.eval_num_workers)
        return self._maybe_prefetch(loader)
