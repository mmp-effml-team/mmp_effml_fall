"""Data parallelism with FSDP2's fully_shard: DDP, FSDP (ZeRO-3) and HSDP are the same call.

The mesh is 2-D (dp_replicate, dp_shard): parameters are replicated over the first dim, sharded over
the second.
  (N, 1) DDP:  every rank holds everything; grads are all-reduced.
  (1, N) FSDP: each rank holds 1/N of params, grads and optimizer state; params are all-gathered for
               forward/backward, grads reduce-scattered.
  (R, S) HSDP: FSDP inside groups of S ranks, DDP across the R groups.
Each transformer block is its own unit (one all-gather per block, overlapped with compute on GPU);
the root unit holds the embeddings, the final norm and the output head.
"""

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard


def apply_fsdp(model, mesh, param_dtype: torch.dtype | None = None):
    # bf16: compute and all-gather in bf16; reduce grads and keep the optimizer's master weights in fp32
    mp = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32)
    for block in model.layers.values():
        fully_shard(block, mesh=mesh, mp_policy=mp)
    fully_shard(model, mesh=mesh, mp_policy=mp)
