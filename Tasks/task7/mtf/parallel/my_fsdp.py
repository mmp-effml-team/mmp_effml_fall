"""FSDP (ZeRO-3) by hand: the homework. torch's FSDP2 (fully_shard) does the same in mtf/parallel/fsdp.py.

fully_shard(module) makes `module` one FSDP unit. Its parameters (except those of units inside it) become shards:
rank r of N keeps rows [r*m/N, (r+1)*m/N) of each parameter, as a DTensor with placement Shard(0). Hooks around
the unit's forward and backward bring the full parameters back only while they are needed:

  pre_forward    all-gather the shards into full parameters (unshard)            part 1.2
  post_forward   free the full parameters, put the shards back (reshard)        part 1.2
  pre_backward   all-gather again: backward needs the parameters too            part 1.3
  post_backward  reshard; reduce-scatter the gradients: rank r gets the average
                 over ranks of its shard of every gradient                      part 1.3

Overlap (part 2): start the all-gather of the next unit before computing this one (prefetch), and do not wait
for reduce-scatters until the end of backward. On GPU, FSDP2 does this with CUDA streams and events. On CPU
with gloo there are no streams: a collective called with async_op=True runs in gloo's own threads and returns a
Work, the main thread keeps computing, and work.wait() blocks until the collective is done.

Units of one model share a Context. apply_fsdp() wraps it like mtf/parallel/fsdp.py: a unit per transformer
block, and the root unit (the whole model) holds what is left: embeddings, final norm, output head.
"""

from dataclasses import dataclass, field
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd import Variable
from torch.distributed.tensor import DTensor, Shard
from torch.profiler import record_function
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten


@dataclass
class Context:
    overlap: bool  # part 2: prefetch all-gathers, wait for reduce-scatters only at the end of backward
    units: list = field(default_factory=list)  # in forward order: root, layers.0, layers.1, ...
    post_forward_order: list = field(default_factory=list)  # units in the order their forward finished, this step
    final_callback_queued: bool = False  # once per backward


class FSDPParam:
    """One parameter of a unit.

    Between forward and backward the module holds `sharded`: an nn.Parameter wrapping a DTensor (Shard(0)) whose
    local tensor is this rank's rows. The optimizer and the checkpoints see only these. While the unit is unsharded
    the module holds `unsharded`, a full-size nn.Parameter in `param_dtype`. It is the same object every time:
    autograd saves it in forward and reads its data in backward. Freeing it keeps the object and drops the data
    (storage resized to 0 bytes); alloc_unsharded() gives the storage back.
    """

    def __init__(self, module: nn.Module, name: str, mesh, param_dtype: torch.dtype | None):
        self.module, self.name, self.mesh = module, name, mesh
        param = module._parameters[name]
        self.full_shape, self.dtype = param.shape, param.dtype  # dtype: of shards, sharded grads, the optimizer
        self.param_dtype = param_dtype or param.dtype  # of the full parameter in forward/backward (bf16: mixed precision)
        n, r = mesh.size(), mesh.get_local_rank()
        if param.shape[0] % n:
            raise ValueError(f"{name}: dim 0 of {tuple(param.shape)} is not divisible by {n} ranks")
        # param may be on "meta" (mtf shards models before their weights exist): the same code must work there.
        # TODO (part 1.1): this rank's rows of `param` as an nn.Parameter wrapping a DTensor on `mesh` with
        # placement Shard(0); keep requires_grad. Only the shard may stay in memory, not the whole param.
        raise NotImplementedError("part 1.1: FSDPParam.__init__")
        self.unsharded: nn.Parameter | None = None  # created on the first unshard, on the device of the shard
        self.to_sharded()

    def to_sharded(self):
        self.module._parameters[self.name] = self.sharded

    def to_unsharded(self):
        self.module._parameters[self.name] = self.unsharded

    def alloc_unsharded(self):
        if self.unsharded is None:
            full = torch.empty(self.full_shape, dtype=self.param_dtype, device=self.sharded.device)
            self.unsharded = nn.Parameter(full, requires_grad=self.sharded.requires_grad)
        storage = self.unsharded.untyped_storage()
        if storage.size() == 0:
            storage.resize_(self.unsharded.numel() * self.unsharded.element_size())

    def free_unsharded(self):
        if self.unsharded is not None:
            self.unsharded.untyped_storage().resize_(0)


class FSDPUnit:
    """The parameters of one module, all-gathered and reduce-scattered together: one collective per unit."""

    def __init__(self, module: nn.Module, mesh, ctx: Context, param_dtype: torch.dtype | None):
        self.module, self.mesh, self.group, self.ctx, self.param_dtype = module, mesh, mesh.get_group(), ctx, param_dtype
        self.params = [FSDPParam(m, name, mesh, param_dtype) for m, name in _own_params(module)]
        self.name = "root"  # apply_fsdp() names the blocks "layers.0", ...
        self.is_unsharded = False
        self.all_gather = None  # (work, output) of the all-gather started by unshard(), until wait_for_unshard()
        self.reduce = None  # part 2.3: (work, output, input) of a reduce-scatter in flight
        self.forward_prefetch: list[FSDPUnit] = []  # part 2.1: units to all-gather when this unit's forward starts
        self.post_forward_index = -1  # position in ctx.post_forward_order this step
        self.needs_post_backward = False  # set by pre_backward, cleared by post_backward

    @torch.no_grad()  # copies and collectives below are not part of the model's autograd graph
    def unshard(self):
        """Start the all-gather of this unit's full parameters (in param_dtype). wait_for_unshard() finishes it."""
        if self.is_unsharded or self.all_gather is not None:  # done already, or in flight (prefetched)
            return
        # TODO (part 1.2): all-gather the shards of all params of the unit with dist.all_gather_into_tensor
        # (one call for the whole unit, or one per param), cast to param_dtype; save (work, output) in self.all_gather.
        # TODO (part 2.1): with self.ctx.overlap, async_op=True: return without waiting.
        raise NotImplementedError("part 1.2: FSDPUnit.unshard")

    @torch.no_grad()
    def wait_for_unshard(self):
        """Finish the all-gather: copy its output into the full parameters and put them into the module."""
        if self.is_unsharded:
            return
        # TODO (part 2.1): wait for the all-gather if it is async.
        # TODO (part 1.2): for each param: alloc_unsharded(), copy its rows from the all-gather output into
        # p.unsharded.data (.data: a plain copy_ would bump the version counter, and backward would refuse
        # the tensors it saved in forward), to_unsharded(). Then drop self.all_gather.
        raise NotImplementedError("part 1.2: FSDPUnit.wait_for_unshard")
        self.is_unsharded = True

    def reshard(self):
        """Free the full parameters, put the shards back into the module."""
        # TODO (part 1.2)
        raise NotImplementedError("part 1.2: FSDPUnit.reshard")
        self.is_unsharded = False

    @torch.no_grad()
    def reduce_grads(self):
        """Reduce-scatter the full gradients (p.unsharded.grad) into this rank's shards, averaged over ranks."""
        # TODO (part 1.3): reduce-scatter the gradients with dist.reduce_scatter_tensor, in each param's dtype
        # (p.dtype: fp32 even when param_dtype is bf16), averaged over ranks. Free the full gradients
        # (p.unsharded.grad = None) and pass the result to accumulate_sharded_grads().
        # TODO (part 2.3): with self.ctx.overlap, async_op=True: save (work, output, input) in self.reduce and
        # return; finish_reduce() completes it. Keep at most one unit's reduce-scatter in flight: finish the
        # previous unit's before starting yours (its input is a full-size gradient buffer).
        raise NotImplementedError("part 1.3: FSDPUnit.reduce_grads")

    @torch.no_grad()
    def accumulate_sharded_grads(self, output: torch.Tensor):
        """Put this rank's reduced gradient shards (`output`, flat, param after param) into p.sharded.grad."""
        # TODO (part 1.3): p.sharded.grad must be a DTensor like p.sharded. A step can have several
        # micro-batches (gradient accumulation): add to the grad if there is one already.
        raise NotImplementedError("part 1.3: FSDPUnit.accumulate_sharded_grads")

    def finish_reduce(self):
        """Part 2.3: wait for the reduce-scatter in flight, if any, and accumulate its result."""
        pass  # TODO (part 2.3)

    def backward_prefetch_target(self):
        """Part 2.2: the unit whose backward comes after this one's, or None if this one is the last."""
        # TODO (part 2.2): use self.post_forward_index and self.ctx.post_forward_order
        return None


def pre_forward(module, args, kwargs):
    unit: FSDPUnit = module.fsdp_unit
    if unit is unit.ctx.units[0]:  # the root: a new forward
        unit.ctx.post_forward_order.clear()
    with record_function(f"FSDP::pre_forward ({unit.name})"):
        unit.unshard()
        unit.wait_for_unshard()
        # TODO (part 2.1): forward prefetch: with unit.ctx.overlap, start the all-gathers of unit.forward_prefetch
    if unit.param_dtype is not None:  # mixed precision: like FSDP2's MixedPrecisionPolicy(cast_forward_inputs=True)
        cast = lambda t: t.to(unit.param_dtype) if torch.is_tensor(t) and t.is_floating_point() else t
        args, kwargs = tree_map(cast, (args, kwargs))
    return register_post_backward(unit, args, kwargs)


def post_forward(module, args, output):
    unit: FSDPUnit = module.fsdp_unit
    with record_function(f"FSDP::post_forward ({unit.name})"):
        unit.reshard()
        unit.post_forward_index = len(unit.ctx.post_forward_order)
        unit.ctx.post_forward_order.append(unit)
    return register_pre_backward(unit, output)


def pre_backward(unit: FSDPUnit, grad: torch.Tensor):
    """Runs when the gradient of the unit's output is ready, before the unit's own backward."""
    if not unit.ctx.final_callback_queued:
        unit.ctx.final_callback_queued = True
        Variable._execution_engine.queue_callback(partial(final_callback, unit.ctx))  # runs after the whole backward
    if unit.needs_post_backward:  # a unit with several outputs: already done
        return grad
    unit.needs_post_backward = True
    with record_function(f"FSDP::pre_backward ({unit.name})"):
        # TODO (part 1.3): backward needs the full parameters
        raise NotImplementedError("part 1.3: pre_backward")
        # TODO (part 2.2): backward prefetch: with unit.ctx.overlap, start the all-gather of
        # unit.backward_prefetch_target()
    return grad


def post_backward(unit: FSDPUnit):
    """Runs when the unit's backward is done: the gradients of its full parameters are ready."""
    unit.needs_post_backward = False
    with record_function(f"FSDP::post_backward ({unit.name})"):
        # TODO (part 1.3)
        raise NotImplementedError("part 1.3: post_backward")


def final_callback(ctx: Context):
    """Runs once at the end of backward."""
    with record_function("FSDP::final_callback"):
        for unit in ctx.units:
            if unit.needs_post_backward:  # its inputs need no grad (the root: token ids), so no post-backward hook ran
                post_backward(unit)
        for unit in ctx.units:
            unit.finish_reduce()
    ctx.final_callback_queued = False


def register_pre_backward(unit: FSDPUnit, output):
    """pre_backward(unit) runs when the gradient w.r.t. an output of the unit is computed."""
    if torch.is_grad_enabled():
        for t in tree_flatten(output)[0]:
            if torch.is_tensor(t) and t.requires_grad:
                t.register_hook(partial(pre_backward, unit))
    return output


def register_post_backward(unit: FSDPUnit, args, kwargs):
    """post_backward(unit) runs when the gradients w.r.t. the unit's inputs are computed, i.e. its backward is done."""
    flat, spec = tree_flatten((args, kwargs))
    idx = [i for i, t in enumerate(flat) if torch.is_tensor(t) and t.requires_grad]
    if not torch.is_grad_enabled() or not idx:
        return args, kwargs
    for i, t in zip(idx, RegisterPostBackward.apply(unit, *[flat[i] for i in idx])):
        flat[i] = t
    return tree_unflatten(flat, spec)


class RegisterPostBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, unit, *inputs):
        ctx.unit = unit
        return inputs

    @staticmethod
    def backward(ctx, *grads):
        post_backward(ctx.unit)
        return (None, *grads)


def _own_params(module: nn.Module):
    """(submodule, name) of every parameter of `module`, except those inside units already made."""
    out = [(module, name) for name, p in module._parameters.items() if p is not None]
    for child in module.children():
        if not hasattr(child, "fsdp_unit"):
            out += _own_params(child)
    return out


def fully_shard(module: nn.Module, mesh, ctx: Context, param_dtype: torch.dtype | None = None) -> FSDPUnit:
    """Make `module` an FSDP unit on a 1-D mesh: shard its parameters and add the hooks."""
    if mesh.ndim != 1:
        raise ValueError(f"fully_shard expects a 1-D mesh, got {mesh}")
    unit = FSDPUnit(module, mesh, ctx, param_dtype)
    module.fsdp_unit = unit
    module.register_forward_pre_hook(pre_forward, with_kwargs=True)
    module.register_forward_hook(post_forward)
    return unit


def apply_fsdp(model, mesh, param_dtype: torch.dtype | None = None, overlap: bool = False):
    """Like mtf/parallel/fsdp.py: a unit per transformer block, then the root unit with the rest of the model."""
    ctx = Context(overlap)
    blocks = []
    for name, block in model.layers.items():
        blocks.append(fully_shard(block, mesh, ctx, param_dtype))
        blocks[-1].name = f"layers.{name}"
    ctx.units = [fully_shard(model, mesh, ctx, param_dtype), *blocks]
    for unit, following in zip(ctx.units, ctx.units[1:]):
        unit.forward_prefetch = [following]
    return model
