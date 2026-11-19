"""End-to-end smoke tests of the training loop on CPU (synthetic data, no downloads)."""

import json
import os

import pytest

from mtf import train
from mtf.config import load_config


def test_debug_config_learns(tmp_path):
    argv = ["--config", "configs/debug.toml", "--train.device=cpu", "--train.steps=60", "--train.log_every=20", f"--train.run_dir={tmp_path}"]
    losses = train.main(argv)
    assert losses[-1] < losses[0] - 1.0, losses
    rows = [json.loads(line) for line in open(tmp_path / "metrics.jsonl")]  # every step, whatever log_every is
    assert len(rows) == 60 and {"loss", "lr", "grad_norm", "tok_per_s", "mem_gb"} <= rows[0].keys()


def test_bf16_autocast_and_profiler(tmp_path):
    argv = ["--config", "configs/debug.toml", "--train.device=cpu", "--train.steps=4", "--train.dtype=bf16",
            f"--train.run_dir={tmp_path}", "--profile.enabled=true", "--profile.wait=1", "--profile.warmup=1", "--profile.active=1"]
    losses = train.main(argv)
    assert all(l == l for l in losses)  # no NaNs
    assert os.path.getsize(tmp_path / "profile" / "trace_rank0.json") > 0
    assert "forward" in open(tmp_path / "profile" / "ops_rank0.txt").read()


def test_config_overrides(tmp_path):
    cfg = load_config(["--train.lr=1e-2", "--config", "configs/debug.toml", "--model.args.n_layers=3", "--profile.enabled=true"])
    assert cfg.train.lr == 1e-2 and cfg.model.args.n_layers == 3 and cfg.profile.enabled  # CLI beats the file
    assert cfg.model.args.dim == 128  # from the file
    with pytest.raises(ValueError, match="unknown config key"):
        load_config(["--train.lrr=1"])
