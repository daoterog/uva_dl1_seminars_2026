"""Train the CIFAR-10 classifier.

python train.py                                   # profile picked from the machine
python train.py --where snellius_mig              # or named explicitly
python train.py --overlay my_experiment.yaml
python train.py data.batch_size=256 train.max_epochs=5
python train.py --print-config                    # show the config and exit
"""

import argparse
import os
import subprocess

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from config import build_config, print_config
from data import CIFAR10DataModule
from module import Classifier


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


def git_commit() -> str:
    """The commit being run: a cluster checkout that was never pulled is a classic waste."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_trainer(cfg: DictConfig) -> L.Trainer:
    """The Trainer, its logger and callbacks, all from the config."""
    logger = WandbLogger(
        project=cfg.logging.project,
        mode=cfg.logging.wandb_mode,
        save_dir=cfg.logging.wandb_dir or ".",
        group=cfg.logging.group,
        tags=list(cfg.logging.tags),
        # The resolved config, so every run records exactly what it trained.
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    callbacks = [
        ModelCheckpoint(dirpath=cfg.logging.checkpoint_dir, monitor="val/acc", mode="max"),
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=cfg.logging.progress_every),
    ]
    return L.Trainer(
        accelerator="auto",
        devices="auto",
        precision=cfg.train.precision,
        max_epochs=cfg.train.max_epochs,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        benchmark=cfg.train.cudnn_benchmark,
        deterministic=cfg.train.deterministic,
        profiler=cfg.train.profiler,
        logger=logger,
        callbacks=callbacks,
    )


def main() -> None:
    """Build the config, print where we are, train, then test the best checkpoint."""
    args, overrides = parse_args()
    cfg = build_config(args.where, args.overlay, overrides)
    if args.print_config:
        print_config(cfg)
        return

    device = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU only"
    print(
        f"commit {git_commit()} | where {cfg.profile.where} | device {device} | "
        f"SLURM job {os.environ.get('SLURM_JOB_ID', '-')}",
        flush=True,
    )
    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision(cfg.train.matmul_precision)

    datamodule = CIFAR10DataModule(cfg.data, seed=cfg.seed)
    model = Classifier(cfg)
    trainer = build_trainer(cfg)
    trainer.fit(model, datamodule=datamodule)
    if not args.no_test:
        trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
