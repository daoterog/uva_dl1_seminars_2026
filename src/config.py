"""Building the run's configuration from layered YAML files.

Merge order, each layer overriding the one before:

    configs/base.yaml  ->  configs/where/<where>.yaml  ->  overlays, in order  ->  key=value

`base.yaml` says *what* is trained. A `where/` profile says how to keep one kind of
machine busy (workers, precision, compile, batch for that GPU). Overlays are experiments
on top, and `key=value` from the command line wins over everything.

The base is put in struct mode before anything is merged into it, so a layer or an
override that names a key base.yaml does not have is an error rather than a silently
ignored typo: `data.num_worker=8` raises instead of training with the default.
"""

import os
from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf, open_dict

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

# SLURM partition -> configs/where/ profile, so a job never runs with the laptop profile.
WHERE_BY_PARTITION = {
    "gpu_mig": "snellius_mig",
    "gpu_a100": "snellius_a100",
}


def detect_where() -> str:
    """Pick the `where/` profile from the SLURM partition, or `local` outside SLURM.

    Raises:
        ValueError: inside a SLURM job on a partition with no profile; better to stop
            than to train a cluster job with the laptop's settings.

    """
    partition = os.environ.get("SLURM_JOB_PARTITION")
    if partition is None:
        return "local"
    if partition not in WHERE_BY_PARTITION:
        raise ValueError(
            f"no configs/where/ profile for SLURM partition {partition!r}; "
            f"pass --where explicitly (known: {sorted(WHERE_BY_PARTITION)})"
        )
    return WHERE_BY_PARTITION[partition]


def build_config(
    where: str | None = None,
    overlays: Sequence[str | Path] = (),
    overrides: Sequence[str] = (),
    config_dir: Path = CONFIG_DIR,
) -> DictConfig:
    """Merge base, profile, overlays and dotlist overrides into one resolved config.

    Args:
        where: Name of a file in `configs/where/`, without `.yaml`. Detected when None.
        overlays: Paths of further YAML files, merged in order.
        overrides: Dotlist overrides such as `data.batch_size=256`.
        config_dir: Directory holding `base.yaml` and `where/`.

    Returns:
        The merged config, with interpolations resolved and a `profile` entry recording
        which layers built it (so every wandb run says what it ran under).

    """
    where = where or detect_where()
    cfg = OmegaConf.load(config_dir / "base.yaml")
    OmegaConf.set_struct(cfg, True)
    layers = [config_dir / "where" / f"{where}.yaml", *map(Path, overlays)]
    for layer in layers:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(layer))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    with open_dict(cfg):
        cfg.profile = {"where": where, "overlays": [str(p) for p in overlays]}
    OmegaConf.resolve(cfg)
    return cfg


def print_config(cfg: DictConfig) -> None:
    """Print the config exactly as the run will see it. Read it before every submit."""
    print(OmegaConf.to_yaml(cfg))
