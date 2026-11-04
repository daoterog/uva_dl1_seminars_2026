from pathlib import Path

import pytest
from omegaconf.errors import ConfigKeyError

from dl1.config import CONFIG_DIR, build_config, detect_where

WHERES = sorted(p.stem for p in (CONFIG_DIR / "where").glob("*.yaml"))


def test_profile_overrides_base_and_is_recorded():
    cfg = build_config("snellius_a100")
    assert cfg.data.num_workers == 17
    assert cfg.optim.lr == 0.1  # untouched by the profile, so base.yaml's value
    assert cfg.profile.where == "snellius_a100"


def test_overlay_then_dotlist_override_in_that_order(tmp_path: Path):
    overlay = tmp_path / "mlp.yaml"
    overlay.write_text("model:\n  name: mlp\ndata:\n  batch_size: 32\n")
    cfg = build_config("local", [overlay], ["data.batch_size=64"])
    assert cfg.model.name == "mlp"  # from the overlay
    assert cfg.data.batch_size == 64  # the command line wins over the overlay's 32
    assert cfg.profile.overlays == [str(overlay)]


def test_typo_in_override_raises():
    with pytest.raises(ConfigKeyError):
        build_config("local", overrides=["data.num_worker=8"])


def test_typo_in_overlay_raises(tmp_path: Path):
    overlay = tmp_path / "typo.yaml"
    overlay.write_text("train:\n  precison: bf16-mixed\n")
    with pytest.raises(ConfigKeyError):
        build_config("local", [overlay])


@pytest.mark.parametrize("where", WHERES)
def test_every_profile_builds(where: str):
    assert build_config(where).profile.where == where


def test_detect_where(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SLURM_JOB_PARTITION", raising=False)
    assert detect_where() == "local"
    monkeypatch.setenv("SLURM_JOB_PARTITION", "gpu_mig")
    assert detect_where() == "snellius_mig"
    monkeypatch.setenv("SLURM_JOB_PARTITION", "rome")
    with pytest.raises(ValueError, match="no configs/where/ profile"):
        detect_where()
