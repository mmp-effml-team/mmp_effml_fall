"""parallelize(): the ONLY place that knows how the model is parallelized.

It runs on the meta-device model, before any weights exist, and turns parameters into DTensors on the
mesh. Each parallelism is one function in its own file: replace the file to swap the implementation.
Here there is one parallelism, data parallel, in two implementations: torch's FSDP2 and yours (the homework).
"""

import torch

from mtf.config import Config
from mtf.parallel import fsdp, my_fsdp


def parallelize(model, mesh, cfg: Config):
    p = cfg.parallel
    param_dtype = torch.bfloat16 if cfg.train.dtype == "bf16" else None  # bf16: mixed precision, see apply_fsdp
    if p.fsdp_impl == "fsdp2":
        fsdp.apply_fsdp(model, mesh["dp_replicate", "dp_shard"], param_dtype, overlap=p.overlap)
    elif p.fsdp_impl == "mine":
        if mesh["dp_replicate"].size() > 1:
            raise ValueError("fsdp_impl=mine shards over one mesh dim: DDP/HSDP (dp_replicate > 1) need fsdp_impl=fsdp2")
        my_fsdp.apply_fsdp(model, mesh["dp_shard"], param_dtype, overlap=p.overlap)
    else:
        raise ValueError(f"unknown parallel.fsdp_impl {p.fsdp_impl!r}: fsdp2 | mine")
    return model
