"""Distributed checks of the hand-written FSDP (mtf/parallel/my_fsdp.py). Run under torchrun (tests/test_my_fsdp.py
does it for you):

    python -m torch.distributed.run --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29511 tests/dist_checks.py forward

Every rank builds the same small model in float64 from one seed and keeps a plain copy as the reference. The
FSDP copy is built like mtf builds models: on meta, sharded, then materialized (to_empty) and filled with the
reference weights. It gets rank r's slice of the batch; its logits and gradient shards must match the reference
on the whole batch. A recorder logs, in order, what the units do (unshard, reshard, ...), the collectives they
call (async or not, dtype) and when those are waited for, and when each block starts computing in forward and in
backward, with the units holding full parameters (or an all-gather result) at that moment: the memory and
overlap checks read this log. Collectives FSDP doesn't need, DTensor redistribution and torch's fully_shard fail.

On failure each rank prints its traceback; under tests/test_my_fsdp.py the first failure (with the lines of
mtf/ it went through) becomes the test's message.
"""

import contextlib
import copy
import datetime
import inspect
import os
import pathlib
import sys
import threading
import time
import traceback

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
import torch.distributed.fsdp as torch_fsdp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from mtf.model.transformer import ModelArgs, Transformer
from mtf.parallel import my_fsdp

ALLOWED = {"all_gather_into_tensor": "all_gather", "all_gather_single": "all_gather", "all_gather": "all_gather",
           "reduce_scatter_tensor": "reduce_scatter", "reduce_scatter_single": "reduce_scatter", "reduce_scatter": "reduce_scatter"}
FORBIDDEN = ["all_reduce", "broadcast", "scatter", "gather", "reduce", "all_to_all", "all_to_all_single", "all_gather_object",
             "send", "recv", "isend", "irecv", "batch_isend_irecv", "all_gather_tensor", "all_gather_tensor_autograd",
             "reduce_scatter_tensor_autograd", "all_reduce_coalesced", "permute_tensor"]
UNIT_METHODS = ["unshard", "wait_for_unshard", "reshard", "reduce_grads", "finish_reduce"]
B, S = 2, 16  # sequences per rank per micro-batch, tokens per sequence


def model_args(tied: bool) -> ModelArgs:  # every dim 0 divisible by 8 ranks
    return ModelArgs(vocab_size=64, dim=32, n_layers=3, n_heads=4, n_kv_heads=2, ffn_dim=64, tie_embeddings=tied)


class Recorder:
    def __init__(self):
        self.events, self.stack, self.local = [], [], threading.local()

    def log(self, kind, **info) -> dict:
        ev = {"kind": kind, "i": len(self.events), "unit": self.stack[-1] if self.stack else None, **info}
        self.events.append(ev)
        return ev

    def find(self, kind, unit=None, before=None, after=-1) -> list[dict]:
        return [e for e in self.events if e["kind"] == kind and (unit is None or e["unit"] == unit)
                and e["i"] > after and (before is None or e["i"] < before)]


class LoggedWork:
    """Wraps the Work an async collective returns, to log when it is waited for."""

    def __init__(self, work, rec: Recorder, ev: dict):
        self._work, self._rec, self._ev = work, rec, ev

    def wait(self, *args, **kwargs):
        if "waited" not in self._ev:
            self._ev["waited"] = self._rec.log(self._ev["kind"] + "_wait", of=self._ev["i"])["i"]
        return self._work.wait(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._work, name)


def allocated(unit) -> bool:
    return any(p.unsharded is not None and p.unsharded.untyped_storage().size() > 0 for p in unit.params)


def _fail(name, why):
    def fail(*args, **kwargs):
        raise AssertionError(f"{name} is not allowed here: {why}")
    return fail


@contextlib.contextmanager
def patched(patches):
    saved = [(obj, name, getattr(obj, name)) for obj, name, _ in patches]
    for obj, name, value in patches:
        setattr(obj, name, value)
    try:
        yield
    finally:
        for obj, name, value in saved:
            setattr(obj, name, value)


def no_fsdp2():
    """torch's fully_shard fails (also where my_fsdp imported it by name)."""
    targets = [(torch_fsdp, "fully_shard")] + [(my_fsdp, n) for n, v in vars(my_fsdp).items() if v is torch_fsdp.fully_shard]
    return patched([(m, n, _fail("torch's fully_shard", "this is your own FSDP")) for m, n in targets])


@contextlib.contextmanager
def instrument(model, rec: Recorder):
    """Log unit methods, collectives and block compute; make collectives FSDP doesn't need fail."""
    units = model.fsdp_unit.ctx.units
    patches = []
    for name in UNIT_METHODS:
        orig = getattr(my_fsdp.FSDPUnit, name)

        def method(self, *args, _orig=orig, _name=name, **kwargs):
            rec.log(_name, of_unit=self.name)
            rec.stack.append(self.name)
            try:
                return _orig(self, *args, **kwargs)
            finally:
                rec.stack.pop()
        patches.append((my_fsdp.FSDPUnit, name, method))
    real = {getattr(c10d, n) for n in ALLOWED}

    def logged(orig, kind):
        sig = inspect.signature(orig)

        def wrapper(*args, **kwargs):
            if getattr(rec.local, "inside", False):  # an alias calling the real op: log once
                return orig(*args, **kwargs)
            a = sig.bind(*args, **kwargs).arguments
            tensor = next(v for v in a.values() if torch.is_tensor(v) or isinstance(v, list))
            dtype = (tensor[0] if isinstance(tensor, list) else tensor).dtype
            ev = rec.log(kind, async_op=bool(a.get("async_op", False)), dtype=dtype)
            rec.local.inside = True
            try:
                work = orig(*args, **kwargs)
            finally:
                rec.local.inside = False
            if work is None:
                ev["waited"] = ev["i"]
                return None
            return LoggedWork(work, rec, ev)
        return wrapper
    for module in (dist, c10d, my_fsdp):
        for name, kind in ALLOWED.items():
            fn = getattr(module, name, None)
            if fn is not None and (module is not my_fsdp or fn in real):
                patches.append((module, name, logged(fn, kind)))
    names = {dist: FORBIDDEN, c10d: FORBIDDEN, funcol: FORBIDDEN + ["reduce_scatter_tensor"]}  # funcol's, not c10d's
    forbidden = {getattr(m, n) for m, ns in names.items() for n in ns if hasattr(m, n)}
    names[my_fsdp] = FORBIDDEN + ["reduce_scatter_tensor"]
    for module, module_names in names.items():
        for name in module_names:
            fn = getattr(module, name, None)
            if fn is not None and (module is not my_fsdp or fn in forbidden):
                patches.append((module, name, _fail(name, "parameters are all-gathered and gradients reduce-scattered "
                                                    "with dist.all_gather_into_tensor / dist.reduce_scatter_tensor")))
    for name in ("redistribute", "full_tensor"):
        patches.append((DTensor, name, _fail(f"DTensor.{name}", "gather and reduce the shards yourself")))

    def snapshot():  # an all-gather result waiting to be copied is a full-size buffer too
        return {u.name for u in units if allocated(u) or u.all_gather is not None}
    handles = []
    for unit in units[1:]:
        block = unit.module

        def on_compute(mod, args, _name=unit.name, _block=block):
            rec.log("compute", block=_name, full=snapshot(), unsharded_dtype=next(_block.parameters()).dtype)

        def on_output(mod, args, out, _name=unit.name):
            def on_grad(grad):  # runs after FSDP's pre_backward hook (registered earlier on the same tensor)
                rec.log("backward", block=_name, full=snapshot())
            if out.requires_grad:
                out.register_hook(on_grad)
        handles += [block.input_layernorm.register_forward_pre_hook(on_compute), block.register_forward_hook(on_output)]
    try:
        with patched(patches):
            yield
    finally:
        for h in handles:
            h.remove()


def rank_info():
    return dist.get_rank(), dist.get_world_size()


def max_error(got, want) -> float:
    return (got.double() - want.double()).abs().max().item() if got.shape == want.shape else float("inf")


def assert_close(name, got, want, tol, hint=""):
    err = max_error(got, want)
    if not err <= tol:
        raise AssertionError(f"rank {dist.get_rank()}: {name}: shape {tuple(got.shape)} vs {tuple(want.shape)}, "
                             f"max abs error {err:.3e} (allowed {tol:.0e}){hint}")


def make(tied=False, dtype=torch.float64, param_dtype=None, overlap=False, mesh=None, meta=True):
    """(reference model, FSDP model, mesh). The reference keeps the full weights; both start equal.

    meta=True: like mtf, the FSDP model is sharded on meta, then materialized and filled with the reference weights
    (the shards' tensors are replaced: a tensor remembered before to_empty is still on meta). meta=False: a copy
    of the reference is sharded (check_shard: the shards must not keep the full weights alive)."""
    torch.manual_seed(0)
    ref = Transformer(model_args(tied)).to(dtype)
    ref.init_weights()
    with torch.no_grad():  # norms not all ones: shards of different ranks differ
        for p in ref.parameters():
            p.add_(0.1 * torch.randn_like(p))
    if meta:
        with torch.device("meta"):
            model = Transformer(model_args(tied)).to(dtype)
    else:
        model = copy.deepcopy(ref)
    mesh = mesh or init_device_mesh("cpu", (dist.get_world_size(),), mesh_dim_names=("dp_shard",))
    with no_fsdp2():
        my_fsdp.apply_fsdp(model, mesh, param_dtype, overlap=overlap)
    assert not any(isinstance(m, torch_fsdp.FSDPModule) for m in model.modules()), "torch's fully_shard was used"
    if meta:
        model.to_empty(device="cpu")
        r, n = rank_info()
        full = dict(ref.named_parameters())
        with torch.no_grad():
            for fqn, p in model.named_parameters():
                local = p.to_local() if isinstance(p, DTensor) else p
                local.copy_(full[fqn].chunk(n)[r] if isinstance(p, DTensor) else full[fqn])
    return ref, model, mesh


def batches(n_micro, dtype=torch.float64):
    g = torch.Generator().manual_seed(1)
    n = dist.get_world_size()
    return [(torch.randint(0, 64, (n * B, S), generator=g), torch.randn(n * B, S, 64, generator=g, dtype=dtype))
            for _ in range(n_micro)]


def my_slice(t):
    r, n = rank_info()
    return t.chunk(n)[r]


def check_rest(model, after: str):
    """Between steps: no full parameters or full gradients kept, the module holds the shards."""
    for unit in model.fsdp_unit.ctx.units:
        assert not allocated(unit), f"after {after}, unit {unit.name} still holds its full parameters"
        assert unit.all_gather is None, f"after {after}, unit {unit.name} keeps its all-gather result (self.all_gather)"
        assert unit.reduce is None, f"after {after}, unit {unit.name} has a reduce-scatter in flight (self.reduce)"
        for p in unit.params:
            assert p.module._parameters[p.name] is p.sharded, f"after {after}, {unit.name}: the module holds not the shard of {p.name}"
            assert p.unsharded is None or p.unsharded.grad is None, f"after {after}, {unit.name}: full grad of {p.name} not freed"


def check_shard():
    r, n = rank_info()
    ref, model, mesh = make(meta=False)  # a real model: a view or the wrong rows would show
    assert model.fsdp_unit.params, "the root unit owns no parameters"
    full = dict(ref.named_parameters())
    for unit in model.fsdp_unit.ctx.units:
        for p in unit.params:
            fqn = next(k for k, v in model.named_parameters() if v is p.sharded)
            s = p.sharded
            assert isinstance(s, torch.nn.Parameter) and isinstance(s, DTensor), f"{fqn}: the shard must be an nn.Parameter wrapping a DTensor"
            assert s.device_mesh == mesh and tuple(s.placements) == (Shard(0),), f"{fqn}: mesh/placements {s.device_mesh}, {s.placements}"
            assert s.shape == full[fqn].shape and s.requires_grad, f"{fqn}: global shape {tuple(s.shape)}, requires_grad {s.requires_grad}"
            local = s.to_local()
            assert_close(f"{fqn}: local shard", local, full[fqn].detach().chunk(n)[r], 0)
            nbytes = local.untyped_storage().nbytes()
            assert nbytes == local.numel() * local.element_size(), f"{fqn}: the shard keeps {nbytes} bytes of storage, not only its own rows"
    with torch.device("meta"):  # mtf shards models on meta, before weights exist
        meta = Transformer(model_args(False))
    with no_fsdp2():
        my_fsdp.apply_fsdp(meta, mesh)
    w = meta.layers["0"].mlp.up_proj.weight
    assert isinstance(w, DTensor) and w.to_local().shape == (64 // n, 32), "sharding a meta model"


def run_steps(model, n_micro, rec=None, train=True):
    """Forward (+ backward) of n_micro micro-batches on this rank's slice; returns the logits."""
    logits = []
    for tokens, w in batches(n_micro, next(model.parameters()).dtype):
        if rec is not None:
            rec.log("micro_batch")
        out = model(my_slice(tokens))
        logits.append(out.detach())
        if train:
            (out * my_slice(w)).sum().backward()  # FSDP averages over ranks: the grad of the whole-batch loss / n
    return logits


def reference_grads(ref, n_micro):
    """Gradients of all micro-batches together and of the last one alone (to name a typical mistake)."""
    n = dist.get_world_size()
    before = {}
    for k, (tokens, w) in enumerate(batches(n_micro, next(ref.parameters()).dtype)):
        if k == n_micro - 1:
            before = {fqn: torch.zeros_like(p) if p.grad is None else p.grad.clone() for fqn, p in ref.named_parameters()}
        ((ref(tokens) * w).sum() / n).backward()
    total = {fqn: p.grad for fqn, p in ref.named_parameters()}
    return total, {fqn: total[fqn] - before[fqn] for fqn in total}


def check_grads(model, ref_grads, tol=1e-10):
    r, n = rank_info()
    total, last = ref_grads
    for fqn, p in model.named_parameters():
        assert p.grad is not None, f"{fqn}: no gradient after backward: the reduce-scatter result never reached p.sharded.grad"
        assert isinstance(p.grad, DTensor) and tuple(p.grad.placements) == (Shard(0),), f"{fqn}: the grad must be a DTensor like the shard"
        assert p.grad.shape == p.shape and p.grad.dtype == p.dtype, f"{fqn}: grad {tuple(p.grad.shape)} {p.grad.dtype}"
        got, want = p.grad.to_local(), total[fqn].chunk(n)[r]
        hint = ""
        if max_error(got, n * want) <= n * tol:
            hint = f"; it is {n} x the expected gradient: summed over ranks, not averaged"
        elif max_error(got, last[fqn].chunk(n)[r]) <= tol:
            hint = "; it is the gradient of the last micro-batch only: overwritten, not accumulated"
        assert_close(f"grad of {fqn}", got, want, tol, hint)


def check_collectives_used(rec, kind):
    names = {e["unit"] for e in rec.find(kind)}
    units = {e["of_unit"] for e in rec.find("unshard" if kind == "all_gather" else "reduce_grads")}
    missing = units - names
    assert not missing, f"no {kind} attributed to units {sorted(missing)}: call the collective inside FSDPUnit methods"


def check_memory(rec, overlap):
    """While a block computes, only the root and that block hold full parameters (+ the prefetched one)."""
    for ev in rec.find("compute") + rec.find("backward"):
        i = int(ev["block"].split(".")[1])
        allowed = {"root", ev["block"]}
        if overlap:
            allowed.add(f"layers.{i + 1 if ev['kind'] == 'compute' else i - 1}")
        extra = ev["full"] - allowed
        assert not extra, (f"while {ev['block']} computes ({ev['kind']}), units {sorted(extra)} hold full parameters "
                           "or an all-gather result" + ("" if overlap else " (overlap=False: no prefetch)"))
        assert ev["block"] in ev["full"], f"{ev['block']} computes ({ev['kind']}) without its full parameters"


def check_no_overlap(rec):
    """overlap=False (part 1): every collective is finished before the next block computes."""
    marks = [e["i"] for e in rec.events if e["kind"] in ("compute", "backward")]
    for kind in ("all_gather", "reduce_scatter"):
        for e in rec.find(kind):
            nxt = next((m for m in marks if m > e["i"]), None)
            assert nxt is None or e.get("waited", nxt) < nxt, (
                f"overlap=False: the {kind} of {e['unit']} is still in flight when the next block computes: "
                "without overlap, wait for each collective right away (async_op=True only with ctx.overlap)")


def check_forward():
    ref, model, _ = make(overlap=False)
    tokens = batches(1)[0][0]
    rec = Recorder()
    with instrument(model, rec), torch.no_grad():
        got = run_steps(model, 1, rec, train=False)[0]
    assert_close("logits (no_grad)", got, ref(my_slice(tokens)).detach(), 1e-12)
    check_rest(model, "forward")
    with instrument(model, rec):
        got = run_steps(model, 1, rec, train=False)[0]
    assert_close("logits", got, ref(my_slice(tokens)).detach(), 1e-12)
    check_rest(model, "forward")
    check_memory(rec, overlap=False)
    check_no_overlap(rec)
    check_collectives_used(rec, "all_gather")


def check_backward(tied=False, overlap=False) -> Recorder:
    ref, model, _ = make(tied=tied, overlap=overlap)
    rec = Recorder()
    with instrument(model, rec):
        run_steps(model, 2, rec)  # two micro-batches: gradients accumulate
    check_rest(model, "backward")  # state first: its messages say more than a gradient mismatch
    check_memory(rec, overlap)
    if not overlap:
        check_no_overlap(rec)
    check_grads(model, reference_grads(ref, 2))
    check_collectives_used(rec, "reduce_scatter")
    return rec


def check_part1_backward():
    check_backward(tied=False)
    check_backward(tied=True)  # the root: tied embeddings are used twice in its forward


def check_mixed():
    """param_dtype=bf16: forward/backward in bf16 on bf16 full params, gradients reduced in fp32. Same as FSDP2."""
    bf16, fp32 = torch.bfloat16, torch.float32
    ref, model, mesh = make(dtype=fp32, param_dtype=bf16)
    fsdp2 = copy.deepcopy(ref)
    mp = torch_fsdp.MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)
    for block in fsdp2.layers.values():
        torch_fsdp.fully_shard(block, mesh=mesh, mp_policy=mp)
    torch_fsdp.fully_shard(fsdp2, mesh=mesh, mp_policy=mp)
    rec = Recorder()
    with instrument(model, rec):
        got = run_steps(model, 2, rec)
    want = run_steps(fsdp2, 2)
    for a, b in zip(got, want):
        assert a.dtype == bf16, f"logits are {a.dtype}, expected bf16"
        assert_close("bf16 logits vs FSDP2", a, b, 0)
    for (fqn, p), q in zip(model.named_parameters(), fsdp2.parameters()):
        assert p.dtype == fp32 and p.grad.dtype == fp32, f"{fqn}: shard {p.dtype}, grad {p.grad.dtype}: both must stay fp32"
        assert_close(f"grad of {fqn} vs FSDP2", p.grad.to_local(), q.grad.to_local(), 1e-6)
    for kind, dtype in (("all_gather", bf16), ("reduce_scatter", fp32)):
        wrong = {str(e["dtype"]) for e in rec.find(kind) if e["dtype"] != dtype}
        assert not wrong, f"{kind} in {sorted(wrong)}, expected {dtype}"
    assert all(e["unsharded_dtype"] == bf16 for e in rec.find("compute")), "blocks compute on non-bf16 full parameters"
    check_rest(model, "backward")


def collective_of(rec, kind, unit, before):
    """The last `kind` collective of `unit` issued before event index `before`."""
    found = rec.find(kind, unit=unit, before=before)
    assert found, f"no {kind} of {unit} before event {before}"
    return found[-1]


def micro_batches(rec):
    starts = [e["i"] for e in rec.find("micro_batch")] + [len(rec.events)]
    return list(zip(starts, starts[1:]))


def check_forward_prefetch():
    rec = check_backward(overlap=True)
    for lo, hi in micro_batches(rec):
        compute = {e["block"]: e["i"] for e in rec.find("compute", after=lo, before=hi)}
        for i in range(len(compute) - 1):
            me, nxt = f"layers.{i}", f"layers.{i + 1}"
            ag = collective_of(rec, "all_gather", nxt, before=compute[nxt])
            assert ag["i"] < compute[me], f"forward: the all-gather of {nxt} starts after {me} starts computing: no prefetch"
            assert ag["async_op"], f"forward: the all-gather of {nxt} is not async (async_op=True)"
            assert ag.get("waited", hi) > compute[me], f"forward: the all-gather of {nxt} is waited for before {me} computes"


def check_backward_prefetch():
    rec = check_backward(overlap=True)
    for lo, hi in micro_batches(rec):
        bwd = {e["block"]: e["i"] for e in rec.find("backward", after=lo, before=hi)}
        for i in range(len(bwd) - 1, 0, -1):
            me, prev = f"layers.{i}", f"layers.{i - 1}"
            ag = collective_of(rec, "all_gather", prev, before=bwd[prev])
            assert ag["i"] < bwd[me], f"backward: the all-gather of {prev} starts after the backward of {me}: no prefetch"
            assert ag["async_op"], f"backward: the all-gather of {prev} is not async (async_op=True)"
            assert ag.get("waited", hi) > bwd[me], f"backward: the all-gather of {prev} is waited for before {me} computes"


def check_async_reduce():
    rec = check_backward(overlap=True)
    for lo, hi in micro_batches(rec):
        bwd = {e["block"]: e["i"] for e in rec.find("backward", after=lo, before=hi)}
        for i in range(len(bwd) - 1, 0, -1):
            me, prev = f"layers.{i}", f"layers.{i - 1}"
            rs = rec.find("reduce_scatter", unit=me, after=bwd[me], before=hi)
            assert rs, f"no reduce-scatter of {me} in its backward"
            assert rs[0]["async_op"], f"the reduce-scatter of {me} is not async (async_op=True)"
            assert rs[0].get("waited", hi) > bwd[prev], f"the reduce-scatter of {me} is waited for before {prev}'s backward starts"
            assert rs[0]["i"] < bwd[prev], f"the reduce-scatter of {me} starts after {prev}'s backward"
        issued = rec.find("reduce_scatter", after=lo, before=hi)
        for k, b in enumerate(issued):  # each unit in flight holds full-size gradient buffers: at most one unit
            for a in issued[:k]:  # (one collective per unit or one per parameter: both fine)
                assert a["unit"] == b["unit"] or a.get("waited", hi) < b["i"], (
                    f"the reduce-scatter of {b['unit']} starts while that of {a['unit']} is still in flight: "
                    "keep at most one unit's reduce-scatter in flight")


CHECKS = {
    "shard": check_shard,
    "forward": check_forward,
    "backward": check_part1_backward,
    "mixed": check_mixed,
    "forward_prefetch": check_forward_prefetch,
    "backward_prefetch": check_backward_prefetch,
    "async_reduce": check_async_reduce,
}

def failure_report(e: BaseException) -> str:
    """The exception with the lines of mtf/ (your code) it went through."""
    root = pathlib.Path(__file__).resolve().parent.parent
    frames = [f for f in traceback.extract_tb(e.__traceback__) if pathlib.Path(f.filename).resolve().is_relative_to(root / "mtf")]
    lines = [f"  {pathlib.Path(f.filename).resolve().relative_to(root)}:{f.lineno} in {f.name}: {(f.line or '').strip()}"
             for f in frames[-6:]]
    return "\n".join([f"rank {dist.get_rank()}, check {sys.argv[1]!r}:", *lines,
                      "".join(traceback.format_exception_only(e)).strip()])


if __name__ == "__main__":
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=60))  # a hang fails in a minute
    try:
        CHECKS[sys.argv[1]]()
    except BaseException as e:
        if out := os.environ.get("FSDP_CHECK_FAILURES"):  # tests/test_my_fsdp.py shows the first one
            pathlib.Path(out, f"rank{dist.get_rank()}.txt").write_text(f"{time.time()}\n{failure_report(e)}")
        raise
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"OK {sys.argv[1]} on {dist.get_world_size()} ranks", flush=True)
    dist.destroy_process_group()
