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


def apply_fsdp(model, mesh, param_dtype: torch.dtype | None = None, overlap: bool = False):
    # bf16: compute and all-gather in bf16; reduce grads and keep the optimizer's master weights in fp32
    mp = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32)
    blocks = [fully_shard(block, mesh=mesh, mp_policy=mp) for block in model.layers.values()]
    fully_shard(model, mesh=mesh, mp_policy=mp)
    if overlap:
        # FSDP2 overlaps with CUDA streams; its all-gathers are blocking calls (async_op=False) issued on a
        # side stream. CPU has no streams: there, a blocking all-gather blocks the training loop. This private
        # switch makes the pre-forward/pre-backward all-gathers async (wait on the returned Work), and explicit
        # forward prefetch starts the next block's all-gather early. Reduce-scatters stay blocking.
        model._set_unshard_async_op(True)
        for unit, following in zip([model, *blocks], blocks):
            unit.set_modules_to_forward_prefetch([following])
