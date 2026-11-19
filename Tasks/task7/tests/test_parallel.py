"""Every parallel layout must reproduce the single-process run: same loss and grad norm at every step.

fp32 on CPU with gloo, so the only differences are float reduction order. Comparing the grad norm (not
just the loss) catches grad-scaling bugs that AdamW would otherwise hide. Each case launches torchrun.
The hand-written FSDP (fsdp_impl=mine) must also match torch's FSDP2 on the same layout.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from mtf import dist, train
from mtf.config import ParallelCfg

ROOT = Path(__file__).resolve().parents[1]
STEPS = 6
GLOBAL_MICRO = 8  # micro_batch * data-parallel size: kept constant so every layout sees the same data
BASE = ["--config", str(ROOT / "configs/debug.toml"), "--train.device=cpu", f"--train.steps={STEPS}", "--train.global_batch=16"]
TIED = ["--model.args.tie_embeddings=true"]  # the output head reuses the embedding: both in the root unit
BF16 = ["--train.dtype=bf16"]

DP_LAYOUTS = {  # name: parallel degrees (unset ones are 1); processes = dp_replicate * dp_shard
    "ddp2": dict(dp_replicate=2),
    "fsdp2": dict(dp_shard=2),
    "hsdp2x2": dict(dp_replicate=2, dp_shard=2),
    "fsdp4": dict(dp_shard=4),
}
MINE = {  # name: (FSDP shards, extra args) for fsdp_impl=mine
    "fsdp4": (4, []),
    "fsdp8_tied": (8, TIED),
}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def rows(run_dir) -> list[dict]:
    return [json.loads(line) for line in open(Path(run_dir) / "metrics.jsonl")]


def torchrun(degrees: dict, args: list[str], run_dir) -> list[dict]:
    d = {"dp_replicate": 1, "dp_shard": 1, **degrees}
    nproc = d["dp_replicate"] * d["dp_shard"]
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}", "--master_addr=127.0.0.1",
           f"--master_port={free_port()}", "-m", "mtf.train", *args, *[f"--parallel.{k}={v}" for k, v in d.items()],
           f"--train.micro_batch={GLOBAL_MICRO // nproc}", f"--train.run_dir={run_dir}"]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONWARNINGS": "ignore", "PYTHONPATH": str(ROOT)}  # this checkout's mtf
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    errors = [line for line in out.stderr.splitlines() if "Error" in line and "torch/distributed/elastic" not in line]
    assert out.returncode == 0, "\n".join(dict.fromkeys(errors[-6:])) or out.stdout[-2000:] + out.stderr[-4000:]
    return rows(run_dir)


def single(args: list[str], run_dir) -> list[dict]:
    train.main(args + [f"--train.micro_batch={GLOBAL_MICRO}", f"--train.run_dir={run_dir}"])
    return rows(run_dir)


def assert_same(got: list[dict], want: list[dict], rel_loss=1e-5, rel_grad_norm=1e-4, what="single process"):
    assert [r["step"] for r in got] == [r["step"] for r in want]
    for g, w in zip(got, want):
        assert g["loss"] == pytest.approx(w["loss"], rel=rel_loss), (f"loss differs from {what}", g, w)
        assert g["grad_norm"] == pytest.approx(w["grad_norm"], rel=rel_grad_norm), (f"grad norm differs from {what}", g, w)


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    return single(BASE, tmp_path_factory.mktemp("reference"))


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Cached reference runs: single process and FSDP2, keyed by (layout, extra args)."""
    cache = {}

    def get(kind: str, shards: int, extra: list[str]):
        key = (kind, shards, tuple(extra))
        if key not in cache:
            run_dir = tmp_path_factory.mktemp(kind)
            cache[key] = single(BASE + extra, run_dir) if kind == "single" else torchrun(dict(dp_shard=shards), BASE + extra, run_dir)
        return cache[key]
    return get


@pytest.mark.parametrize("layout", DP_LAYOUTS)
def test_layout_matches_single_process(reference, tmp_path, layout):
    assert_same(torchrun(DP_LAYOUTS[layout], BASE, tmp_path), reference)


def check_mine(runs, tmp_path, layout, overlap):
    shards, extra = MINE[layout]
    got = torchrun(dict(dp_shard=shards), BASE + extra + ["--parallel.fsdp_impl=mine", f"--parallel.overlap={overlap}"], tmp_path)
    assert_same(got, runs("single", 1, extra))
    assert_same(got, runs("fsdp2", shards, extra), rel_loss=1e-6, rel_grad_norm=1e-6, what="FSDP2")


@pytest.mark.part1
@pytest.mark.parametrize("layout", MINE)
def test_mine_matches_single_process_and_fsdp2(runs, tmp_path, layout):
    check_mine(runs, tmp_path, layout, overlap=False)


@pytest.mark.part2
@pytest.mark.parametrize("layout", MINE)
def test_mine_with_overlap_matches_single_process_and_fsdp2(runs, tmp_path, layout):
    check_mine(runs, tmp_path, layout, overlap=True)


@pytest.mark.part1
def test_mine_bf16_matches_fsdp2(runs, tmp_path):
    got = torchrun(dict(dp_shard=4), BASE + BF16 + ["--parallel.fsdp_impl=mine", "--parallel.overlap=false"], tmp_path)
    assert_same(got, runs("fsdp2", 4, BF16), rel_loss=1e-4, rel_grad_norm=1e-3, what="FSDP2 in bf16")


def test_resume_with_a_different_layout(reference, tmp_path):
    saved = tmp_path / "saved"
    assert_same(torchrun(DP_LAYOUTS["fsdp4"], BASE + ["--ckpt.every=3"], saved), reference)  # saves steps 3 and 6
    shutil.rmtree(saved / "checkpoints" / "step_6")
    for name in ("ddp2", "single"):  # resume from step 3 with another layout; steps 4-6 must match
        run_dir = tmp_path / name
        shutil.copytree(saved / "checkpoints", run_dir / "checkpoints")
        args = BASE + ["--ckpt.resume=true"]
        assert_same(single(args, run_dir) if name == "single" else torchrun(DP_LAYOUTS[name], args, run_dir), reference[3:])


def test_bf16_mixed_precision(reference, tmp_path):
    got = torchrun(DP_LAYOUTS["fsdp2"], BASE + BF16, tmp_path)
    assert got[0]["loss"] == pytest.approx(reference[0]["loss"], rel=1e-2)  # same weights, bf16 compute
    assert got[-1]["loss"] < got[0]["loss"]


def test_bad_layouts_fail_loudly():
    with pytest.raises(ValueError, match="torchrun"):
        dist.init("cpu", parallel=ParallelCfg(dp_shard=2))  # degrees > 1 in a single process
    with pytest.raises(ValueError, match="multiply"):
        dist.build_mesh("cpu", 4, ParallelCfg(dp_replicate=3))
