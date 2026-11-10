"""Distributed checks of the context-parallel pieces. Run under torchrun (tests/test_cp.py does it for you):

    python -m torch.distributed.run --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29511 tests/dist_checks.py ring

Every rank builds the same inputs from one seed, computes the single-process answer itself, and compares its
part of the distributed result with it. float64, so the tolerance is tight. Collectives that a part must not
use are replaced by functions that fail, both in torch and in mtf.parallel.cp / mtf.parallel.comm.
"""

import contextlib
import datetime
import sys

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
import torch.nn.functional as F

from mtf.parallel import comm, cp

A2A = ["all_to_all", "all_to_all_single", "all_to_all_single_autograd"]
COLLECTIVES = A2A + ["all_gather", "all_gather_into_tensor", "all_gather_object", "all_gather_tensor",
                     "all_gather_tensor_autograd", "broadcast", "scatter", "gather", "reduce", "all_reduce",
                     "reduce_scatter", "reduce_scatter_tensor", "reduce_scatter_tensor_autograd", "permute_tensor"]
B, H, D = 2, 8, 16


@contextlib.contextmanager
def forbid(names: list[str], why: str):
    """Make torch's collectives `names` fail (also where cp.py / comm.py imported them by name)."""
    builtins = {getattr(m, n) for m in (dist, c10d, funcol) for n in names if hasattr(m, n)}
    saved = []
    for module in (dist, c10d, funcol, cp, comm):
        for name in names:
            fn = getattr(module, name, None)
            if fn is not None and (module in (dist, c10d, funcol) or fn in builtins):
                saved.append((module, name, fn))
                setattr(module, name, _failing(name, why))
    try:
        yield
    finally:
        for module, name, fn in saved:
            setattr(module, name, fn)


def _failing(name, why):
    def fail(*args, **kwargs):
        raise AssertionError(f"{name} is not allowed here: {why}")
    return fail


def shard(x, rank, size, dim):  # the layout of part 1, written here so these checks don't depend on it
    return x.chunk(size, dim=dim)[rank]


def assert_close(name, got, want):
    err = (got - want).abs().max().item() if got.shape == want.shape else float("inf")
    if not err < 1e-9:
        raise AssertionError(f"rank {dist.get_rank()}: {name} differs from single-process torch: "
                             f"shape {tuple(got.shape)} vs {tuple(want.shape)}, max abs error {err:.3e}")


def check_attention(fn, forbidden: list[str], why: str):
    r, n = dist.get_rank(), dist.get_world_size()
    s = 64  # divisible by n = 4, 8
    g = torch.Generator().manual_seed(0)
    q, k, v, w = (torch.randn(B, H, s, D, generator=g, dtype=torch.float64) for _ in range(4))
    ref = [t.clone().requires_grad_() for t in (q, k, v)]
    out = F.scaled_dot_product_attention(*ref, is_causal=True)
    (out * w).sum().backward()

    local = [shard(t, r, n, 2).requires_grad_() for t in (q, k, v)]
    positions = shard(torch.arange(s).expand(B, s), r, n, 1)
    with forbid(forbidden, why):
        local_out = fn(*local, positions, dist.group.WORLD)
        assert_close("forward output", local_out, shard(out, r, n, 2))
        (local_out * shard(w, r, n, 2)).sum().backward()
    for name, got, want in zip(("dq", "dk", "dv"), local, ref):
        assert_close(name, got.grad, shard(want.grad, r, n, 2))


def a2a_cases(n: int, uneven: bool):
    """Per rank: (input, output_split_sizes, input_split_sizes, group) for WORLD and two subgroups."""
    r = dist.get_rank()
    sub = [dist.new_group(list(range(i, n, 2))) for i in (0, 1)]  # even and odd ranks; every rank creates both
    cases = []
    for group in (dist.group.WORLD, sub[r % 2]):
        m, gr = dist.get_world_size(group), dist.get_rank(group)
        g = torch.Generator().manual_seed(7 + m)
        rows = torch.randint(0, 4, (m, m), generator=g) if uneven else torch.full((m, m), 3)  # rows[i, j]: i -> j
        if uneven:
            rows[0, 1] = rows[1, 0] = 2  # some traffic between ranks for sure
            if m > 2:
                rows[0, m - 1] = rows[m - 1, 0] = 0  # and some empty chunks
        x = torch.randn(int(rows[gr].sum()), 5, 2, generator=torch.Generator().manual_seed(100 + r), dtype=torch.float64)
        cases.append((x, rows[:, gr].tolist(), rows[gr].tolist(), group))
    return cases


def reference_a2a(x, out_splits, in_splits, group):
    out = x.new_empty((sum(out_splits), *x.shape[1:]))
    dist.all_to_all_single(out, x, out_splits, in_splits, group=group)
    return out


def check_a2a(uneven: bool, grad: bool):
    n = dist.get_world_size()
    for x, out_splits, in_splits, group in a2a_cases(n, uneven):
        want = reference_a2a(x, out_splits, in_splits, group)
        if not uneven:  # equal splits: also the None (= equal) default
            with forbid(COLLECTIVES, "build all-to-all from P2P ops"):
                assert_close("all_to_all_single(splits=None)", comm.all_to_all_single(x, group=group), want)
        if not grad:
            with forbid(COLLECTIVES, "build all-to-all from P2P ops"):
                assert_close("all_to_all_single", comm.all_to_all_single(x, out_splits, in_splits, group), want)
            continue
        w = torch.randn(want.shape, generator=torch.Generator().manual_seed(200 + dist.get_rank()), dtype=torch.float64)
        want_grad = reference_a2a(w, in_splits, out_splits, group)  # the adjoint: rows go back where they came from
        xg = x.clone().requires_grad_()
        with forbid(COLLECTIVES, "build all-to-all from P2P ops"):
            y = comm.all_to_all(xg, out_splits, in_splits, group)
            assert_close("all_to_all", y, want)
            (y * w).sum().backward()
        assert_close("all_to_all grad", xg.grad, want_grad)


CHECKS = {
    "allgather": lambda: check_attention(cp.allgather_attention, [], ""),
    "ring": lambda: check_attention(cp.ring_attention, COLLECTIVES, "ring attention passes K/V with P2P ops"),
    "ulysses": lambda: check_attention(cp.ulysses_attention, A2A, "use your all-to-all from mtf/parallel/comm.py"),
    "a2a_equal": lambda: check_a2a(uneven=False, grad=False),
    "a2a_uneven": lambda: check_a2a(uneven=True, grad=False),
    "a2a_grad": lambda: check_a2a(uneven=True, grad=True),
}

if __name__ == "__main__":
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=60))  # a hang fails in a minute
    CHECKS[sys.argv[1]]()
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"OK {sys.argv[1]} on {dist.get_world_size()} ranks", flush=True)
    dist.destroy_process_group()
