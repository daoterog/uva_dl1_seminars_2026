"""The LightningModule: one model, its loss, its metrics, its optimiser."""

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric.utilities.throughput import measure_flops
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy

from dl1.metrics import ExpectedCalibrationError, PerClassAccuracy
from dl1.models import build_model

CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


class Classifier(L.LightningModule):
    """Image classifier on CIFAR-10, configured entirely from the run's config."""

    def __init__(self, cfg: DictConfig | dict) -> None:
        """Build the model and one set of metrics per split.

        Args:
            cfg: The whole run config. Saved as a plain dict in the checkpoint, so
                `Classifier.load_from_checkpoint(path)` rebuilds the same model.

        """
        super().__init__()
        cfg = OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg
        self.save_hyperparameters({"cfg": OmegaConf.to_container(cfg, resolve=True)})
        self.cfg = cfg
        self.net = build_model(cfg.model)
        if cfg.train.compile:
            # In place: the module stays the same object, so state_dict keys are unchanged
            # (torch.compile(model) would return a wrapper whose keys start `_orig_mod.`).
            self.net.compile(mode=cfg.train.compile_mode)

        n = cfg.model.num_classes
        self.train_acc = MulticlassAccuracy(num_classes=n)
        # One collection per evaluation split: the same metric object must never be
        # updated by two splits, or each would report the other's samples too.
        self.eval_metrics = nn.ModuleDict(
            {
                split: nn.ModuleDict(
                    {
                        "loss": MeanMetric(),
                        "acc": MulticlassAccuracy(num_classes=n),
                        "ece": ExpectedCalibrationError(num_bins=cfg.metrics.ece_bins),
                        "per_class": PerClassAccuracy(num_classes=n),
                    }
                )
                for split in ("val", "test")
            }
        )
        # Read by Lightning's ThroughputMonitor to report model FLOP utilisation (MFU).
        self.flops_per_batch = self._count_flops(cfg)

    @staticmethod
    def _count_flops(cfg: DictConfig) -> int:
        """FLOPs of one training step (forward + backward) at the training batch size.

        Counted on a fresh, uncompiled copy on the meta device: no memory, no compute,
        and inductor's fused kernels would be invisible to the counter.
        """
        with torch.device("meta"):
            net = build_model(cfg.model)
            x = torch.empty(cfg.data.batch_size, 3, 32, 32)
        return measure_flops(net, lambda: net(x), loss_fn=torch.Tensor.sum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 32, 32) images -> (B, num_classes) logits."""
        return self.net(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], _idx: int) -> torch.Tensor:
        """Cross-entropy on one batch; metrics updated on the GPU, logged per epoch."""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=self.cfg.optim.label_smoothing)
        self.train_acc.update(logits, y)
        # Under DDP this is each process's own loss. sync_dist=True would make it global,
        # at the price of an all-reduce on every step; the validation metrics are global.
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(y))
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)
        self._antipatterns(loss, logits, y)
        return loss

    def _antipatterns(self, loss: torch.Tensor, logits: torch.Tensor, y: torch.Tensor) -> None:
        """The per-step host syncs of configs/failures/03_host_syncs.yaml. Off by default."""
        ap = self.cfg.antipatterns
        if ap.item_every_step:
            self.last_loss = float(loss)
        if ap.cpu_metric_every_step:
            predictions = logits.argmax(dim=-1).cpu().numpy()
            self.last_acc = float(np.mean(predictions == y.cpu().numpy()))
        if ap.check_finite_every_step and not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {self.global_step}")

    def _eval_step(self, batch: tuple[torch.Tensor, torch.Tensor], split: str) -> None:
        x, y = batch
        logits = self(x)
        metrics = self.eval_metrics[split]
        # Weighted by the batch size, so a short last batch counts for what it holds.
        metrics["loss"].update(F.cross_entropy(logits, y), weight=len(y))
        for name in ("acc", "ece", "per_class"):
            metrics[name].update(logits, y)

    def _log_eval_epoch(self, split: str) -> None:
        """Compute (which syncs across processes), log, and reset one split's metrics."""
        metrics = self.eval_metrics[split]
        self.log_dict(
            {f"{split}/{name}": metrics[name].compute() for name in ("loss", "acc", "ece")},
            prog_bar=split == "val",
        )
        per_class = metrics["per_class"].compute()
        self.log_dict({f"{split}/acc_{c}": per_class[i] for i, c in enumerate(CLASSES)})
        for metric in metrics.values():
            metric.reset()

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], _idx: int) -> None:
        """Update the validation metrics."""
        self._eval_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        """Log the validation metrics over the whole split."""
        self._log_eval_epoch("val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], _idx: int) -> None:
        """Update the test metrics."""
        self._eval_step(batch, "test")

    def on_test_epoch_end(self) -> None:
        """Log the test metrics over the whole split."""
        self._log_eval_epoch("test")

    def configure_optimizers(self) -> dict:
        """SGD with Nesterov momentum and a one-cycle LR schedule over the whole run."""
        o = self.cfg.optim
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=o.lr,
            momentum=o.momentum,
            weight_decay=o.weight_decay,
            nesterov=o.nesterov,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=o.lr, total_steps=self.trainer.estimated_stepping_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
