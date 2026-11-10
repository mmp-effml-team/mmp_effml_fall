"""parallelize(): the ONLY place that knows the order parallelisms are applied in.

It runs on the meta-device model, before any weights exist, and turns parameters into DTensors on the
mesh. Each parallelism is one function in its own file: replace the file to swap the implementation.
The train loop also asks this module which part of a batch this rank keeps (context parallel).
"""

import torch

from mtf.config import Config
from mtf.parallel.cp import apply_cp, cp_shard
from mtf.parallel.fsdp import apply_fsdp


def parallelize(model, mesh, cfg: Config):
    # Order: context parallel attention -> FSDP (fully_shard must come last: it wraps the final modules).
    a, p = model.args, cfg.parallel
    if p.cp > 1:
        if cfg.data.seq_len % p.cp:
            raise ValueError(f"seq_len={cfg.data.seq_len} must be divisible by cp={p.cp}")
        if p.cp_attn == "ulysses" and a.n_heads % p.cp:
            raise ValueError(f"ulysses: cp={p.cp} must divide n_heads={a.n_heads}")
        apply_cp(model, mesh["cp"], p.cp_attn)
    # Which ranks must average their gradients? FSDP reduces (and shards) over the mesh it is given.
    # TODO (part 1): with cp > 1, the ranks of a cp group compute grads on different tokens of the same
    # sequences. Which mesh dims must FSDP reduce gradients over then?
    fsdp_mesh = mesh["dp_replicate", "dp_shard"]
    param_dtype = torch.bfloat16 if cfg.train.dtype == "bf16" else None
    apply_fsdp(model, fsdp_mesh, param_dtype)
    return model


def shard_batch(batch: dict, mesh) -> dict:
    """Context parallel: keep this rank's part of every sequence (tokens, labels, positions). No-op if cp=1."""
    if mesh is None or mesh["cp"].size() == 1:
        return batch
    rank, size = mesh["cp"].get_local_rank(), mesh["cp"].size()
    return {k: cp_shard(v, rank, size) for k, v in batch.items()}
