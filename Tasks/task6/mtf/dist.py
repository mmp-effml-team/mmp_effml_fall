"""Runtime: the ONLY module that knows about devices, backends and process groups.

cuda -> nccl | cpu -> gloo | mps -> single process only
(gloo can't move MPS tensors, and DeviceMesh("mps") fails in torch 2.15; multi-process on a Mac uses CPU).
torch.distributed is initialized only when launched by torchrun with more than one process. A single
process gets no mesh and trains a plain (non-DTensor) model: that run is the reference the parallel
layouts are tested against.
"""

import math
import os
import random
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor

from mtf.config import ParallelCfg


@dataclass
class Runtime:
    device: torch.device
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    mesh: DeviceMesh | None = None  # None: single process

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def dp_rank(self) -> int:  # which slice of the global batch this rank reads
        return self.mesh["dp"].get_local_rank() if self.mesh is not None else 0

    @property
    def dp_size(self) -> int:
        return self.mesh["dp"].size() if self.mesh is not None else 1


def init(device: str = "auto", seed: int = 0, parallel: ParallelCfg | None = None) -> Runtime:
    parallel = parallel or ParallelCfg()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank, local_rank = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available() and world_size == 1:
            device = "mps"
        else:
            device = "cpu"
    if device == "mps" and world_size > 1:
        raise ValueError("mps supports a single process only; use --train.device=cpu with torchrun")
    dev = torch.device("cuda", local_rank) if device == "cuda" else torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    mesh = None
    if world_size > 1:
        dist.init_process_group("nccl" if dev.type == "cuda" else "gloo", device_id=dev if dev.type == "cuda" else None)
        mesh = build_mesh(dev.type, world_size, parallel)
    elif any(getattr(parallel, k) > 1 for k in ("dp_replicate", "dp_shard", "cp")):
        raise ValueError("parallel degrees > 1 need several processes: launch with torchrun --nproc_per_node=N")
    random.seed(seed)
    torch.manual_seed(seed)
    return Runtime(dev, rank, world_size, local_rank, mesh)


def build_mesh(device_type: str, world_size: int, p: ParallelCfg) -> DeviceMesh:
    """One mesh for every parallelism, dims ordered outer -> inner: (dp_replicate, dp_shard, cp).

    Size-1 dims are kept, so code can always ask for mesh["cp"]. cp is the innermost dim: the ranks of one
    cp group are neighbours (on a GPU cluster: inside one node, where communication is fastest).
    """
    dims = {"dp_replicate": p.dp_replicate, "dp_shard": p.dp_shard, "cp": p.cp}
    if dims["dp_shard"] == -1:
        dims["dp_shard"] = world_size // math.prod(v for k, v in dims.items() if k != "dp_shard")
    if math.prod(dims.values()) != world_size:
        raise ValueError(f"parallel degrees {dims} multiply to {math.prod(dims.values())}, not the world size {world_size}")
    mesh = init_device_mesh(device_type, tuple(dims.values()), mesh_dim_names=tuple(dims))
    mesh["dp_replicate", "dp_shard"]._flatten("dp")  # data loading: ranks in one dp group read different data
    mesh["dp_shard", "cp"]._flatten("dp_shard_cp")  # for FSDP under context parallel (see parallelize)
    return mesh


def mean(t: torch.Tensor) -> torch.Tensor:
    """Average a tensor over all ranks, in place (for logging the loss)."""
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t


def full(t: torch.Tensor) -> torch.Tensor:
    """A plain tensor from a tensor or a DTensor."""
    return t.full_tensor() if isinstance(t, DTensor) else t


def item(t: torch.Tensor) -> float:
    """Python float from a tensor or a DTensor (e.g. the grad norm of a sharded model)."""
    return full(t).item()


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


def print0(*args, **kwargs):
    if is_main():
        print(*args, **kwargs, flush=True)


def destroy():
    if dist.is_initialized():
        dist.destroy_process_group()
