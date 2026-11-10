"""Unit tests of the context-parallel pieces, parts 1-4 (grading: tools/grade.py).

The distributed ones launch torchrun with N gloo processes running tests/dist_checks.py, which compares each
rank's result with single-process torch on the same inputs, forward and backward.
"""

import os
import subprocess
import sys

import pytest
import torch

from mtf.parallel.cp import cp_shard
from tests.test_parallel import ROOT, free_port


def run_check(name: str, n: int):
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={n}", "--master_addr=127.0.0.1",
           f"--master_port={free_port()}", str(ROOT / "tests" / "dist_checks.py"), name]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONWARNINGS": "ignore", "PYTHONPATH": str(ROOT)}  # this checkout's mtf
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    errors = [line for line in out.stderr.splitlines() if "Error" in line and "torch/distributed/elastic" not in line]
    assert out.returncode == 0, "\n".join(errors[-6:]) or out.stderr[-3000:]


@pytest.mark.part1
def test_cp_shard():
    x = torch.arange(16).view(1, 16)
    assert cp_shard(x, 0, 2).tolist() == [[0, 1, 2, 3, 4, 5, 6, 7]]
    assert cp_shard(x, 1, 2).tolist() == [[8, 9, 10, 11, 12, 13, 14, 15]]
    parts = [cp_shard(x, r, 4) for r in range(4)]
    assert [p.tolist() for p in parts] == [[list(range(4 * r, 4 * r + 4))] for r in range(4)]
    y = torch.randn(3, 5, 32, 2)  # any dim; the other dims untouched
    parts = [cp_shard(y, r, 8, dim=2) for r in range(8)]
    assert all(p.shape == (3, 5, 4, 2) for p in parts)
    assert torch.equal(torch.cat(parts, dim=2), y)


@pytest.mark.part1
@pytest.mark.parametrize("n", [4, 8])
def test_allgather_attention(n):
    run_check("allgather", n)


@pytest.mark.part2
@pytest.mark.parametrize("n", [4, 8])
def test_ring_attention(n):
    run_check("ring", n)


@pytest.mark.part3
@pytest.mark.parametrize("n", [4, 8])
def test_a2a_equal_splits(n):
    run_check("a2a_equal", n)


@pytest.mark.part3
@pytest.mark.parametrize("n", [4, 8])
def test_a2a_uneven_splits_and_subgroups(n):
    run_check("a2a_uneven", n)


@pytest.mark.part3
@pytest.mark.parametrize("n", [4, 8])
def test_a2a_autograd(n):
    run_check("a2a_grad", n)


@pytest.mark.part4
@pytest.mark.parametrize("n", [4, 8])
def test_ulysses_attention(n):
    run_check("ulysses", n)
