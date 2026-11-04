"""Train the CIFAR-10 classifier.

python train.py                                   # profile picked from the machine
python train.py --where snellius_mig              # or named explicitly
python train.py --overlay my_experiment.yaml
python train.py optim.lr=0.05 train.max_epochs=5
python train.py --print-config                    # show the config and exit
"""

import argparse

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from src.config import build_config, print_config
from src.data import CIFAR10DataModule
from src.pl_module import Classifier


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Flags, plus the remaining `key=value` tokens as dotlist overrides."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--where", help="a configs/where/ profile (default: detected)")
    parser.add_argument(
        "--overlay", action="append", default=[], help="extra YAML, merged in order"
    )
    parser.add_argument("--print-config", action="store_true", help="print the config and exit")
    parser.add_argument("--no-test", action="store_true", help="skip the test-set evaluation")
    return parser.parse_known_args()


def device_name() -> str:
    """The device this run will train on, as Lightning's accelerator="auto" will pick it."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name()
    if torch.backends.mps.is_available():
        return "Apple MPS"
    return "CPU"


def build_trainer(cfg: DictConfig) -> L.Trainer:
    """The Trainer, its logger and its callbacks, all from the config."""
    log = cfg.logging
    logger = WandbLogger(
        entity=log.entity,
        project=log.project,
        name=log.name,
        group=log.group,
        tags=list(log.tags),
        # The resolved config, so every run records exactly what it trained.
        config=OmegaConf.to_container(cfg, resolve=True),
        offline=log.offline,
        save_dir=log.save_dir or ".",
    )
    callbacks = [
        # No dirpath: the checkpoints go into this run's own directory.
        ModelCheckpoint(monitor="val/acc", mode="max"),
        LearningRateMonitor(logging_interval="step"),
    ]
    t = cfg.train
    return L.Trainer(
        max_epochs=t.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=t.precision,
        gradient_clip_val=t.gradient_clip_val,
        accumulate_grad_batches=t.accumulate_grad_batches,
        check_val_every_n_epoch=t.check_val_every_n_epoch,
        log_every_n_steps=log.log_every_n_steps,
        logger=logger,
        callbacks=callbacks,
        fast_dev_run=t.fast_dev_run,
        profiler=t.profiler,
    )


def main() -> None:
    """Build the config, train, then test the best checkpoint."""
    args, overrides = parse_args()
    cfg = build_config(args.where, args.overlay, overrides)
    if args.print_config:
        print_config(cfg)
        return

    print(f"where {cfg.profile.where} | device {device_name()}", flush=True)
    L.seed_everything(cfg.seed, workers=True)

    datamodule = CIFAR10DataModule(cfg.data, seed=cfg.seed)
    model = Classifier(cfg)
    trainer = build_trainer(cfg)
    trainer.fit(model, datamodule=datamodule)
    if not args.no_test:
        # fast_dev_run keeps no checkpoints: test the weights as they are.
        ckpt_path = None if cfg.train.fast_dev_run else "best"
        trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
