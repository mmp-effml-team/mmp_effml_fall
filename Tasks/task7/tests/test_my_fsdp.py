"""Unit tests of the hand-written FSDP, parts 1-2 (grading: tools/grade.py).

Each launches torchrun with N gloo processes running tests/dist_checks.py, which compares the FSDP model with a
plain copy on the same data (forward and backward, float64) and checks memory and the order of collectives.
"""

import os
import pathlib
import subprocess
import sys
import tempfile

import pytest

from tests.test_parallel import ROOT, free_port

N = pytest.mark.parametrize("n", [4, 8])


def run_check(name: str, n: int):
    """Run one check of dist_checks.py on n ranks; on failure show the first rank's error and its lines in mtf/."""
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={n}", "--master_addr=127.0.0.1",
           f"--master_port={free_port()}", str(ROOT / "tests" / "dist_checks.py"), name]
    with tempfile.TemporaryDirectory() as failures:
        env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONWARNINGS": "ignore", "PYTHONPATH": str(ROOT),  # this checkout's mtf
               "FSDP_CHECK_FAILURES": failures}
        out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
        reports = sorted(pathlib.Path(failures).glob("rank*.txt"), key=lambda p: float(p.read_text().split("\n", 1)[0]))
        first = reports[0].read_text().split("\n", 1)[1] if reports else None  # the others usually fail because of it
    if out.returncode != 0:
        errors = [line for line in out.stderr.splitlines() if "Error" in line and "torch/distributed/elastic" not in line]
        pytest.fail(first or "\n".join(dict.fromkeys(errors[-8:])) or out.stderr[-3000:], pytrace=False)


@pytest.mark.part1
@N
def test_shard(n):
    run_check("shard", n)


@pytest.mark.part1
@N
def test_forward(n):
    run_check("forward", n)


@pytest.mark.part1
@N
def test_backward(n):
    run_check("backward", n)


@pytest.mark.part1
@N
def test_mixed_precision(n):
    run_check("mixed", n)


@pytest.mark.part2
@N
def test_forward_prefetch(n):
    run_check("forward_prefetch", n)


@pytest.mark.part2
@N
def test_backward_prefetch(n):
    run_check("backward_prefetch", n)


@pytest.mark.part2
@N
def test_async_reduce_scatter(n):
    run_check("async_reduce", n)
