"""Context parallel (CP): rank r of a cp group holds the r-th of cp contiguous chunks of every sequence.

Everything except attention works token by token, so it runs on the local tokens unchanged. Attention needs
the keys and values of the whole sequence; the three functions below get them in different ways:
  allgather_attention  all-gather K and V                                                    (part 1)
  ring_attention       pass K/V blocks around a ring with P2P ops, merge them with online softmax (part 2)
  ulysses_attention    all-to-all from sequence-sharded to head-sharded, full attention, back  (part 4)
All three take q, k, v [B, H, S_local, D] and positions [B, S_local] (the absolute positions of this rank's
tokens) and return this rank's attention output [B, H, S_local, D], the rows of the full causal attention
that belong to its tokens. A query attends to a key iff q_pos >= k_pos (causal_mask).
"""

import math
from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._functional_collectives import all_gather_tensor, all_gather_tensor_autograd

from mtf.parallel.comm import all_to_all


def apply_cp(model, mesh, impl: str):
    fn = {"allgather": allgather_attention, "ring": ring_attention, "ulysses": ulysses_attention}[impl]
    for block in model.layers.values():
        block.self_attn.attn_fn = partial(_with_all_kv_heads, fn, group=mesh.get_group())


def _with_all_kv_heads(fn, q, k, v, positions, group):
    """Grouped-query attention: repeat K/V heads to match Q, so the functions below see equal head counts."""
    if k.shape[1] != q.shape[1]:
        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    return fn(q, k, v, positions, group)


def causal_mask(q_pos, k_pos):
    """q_pos [B, Sq], k_pos [B, Sk] -> bool [B, 1, Sq, Sk]: True where the query may attend to the key."""
    return q_pos[:, None, :, None] >= k_pos[:, None, None, :]


def cp_shard(x: torch.Tensor, rank: int, size: int, dim: int = 1) -> torch.Tensor:
    """Part 1. This rank's part of the sequence dim `dim` of x.

    Cut dim into `size` equal contiguous chunks; rank r keeps chunk r. Example, size=2, x = [0 1 2 3 4 5 6 7]:
    rank 0 keeps [0 1 2 3], rank 1 keeps [4 5 6 7].
    """
    raise NotImplementedError("part 1: cp_shard")


def allgather_attention(q, k, v, positions, group):
    """Part 1. Gather K, V and their positions from every rank of `group`, attend with the local queries.

    The K/V gather must be differentiable: in backward, each rank gets the grads of its own K/V back.
    Use torch.distributed._functional_collectives (torch.distributed.nn.functional gives wrong grads on gloo).
    """
    raise NotImplementedError("part 1: allgather_attention")


def ring_attention(q, k, v, positions, group):
    """Part 2. Ring attention: K/V blocks travel around the ring of `group` ranks; only P2P ops are allowed.

    Step i (i = 0..n-1): attend to the block that came from rank r - i, then pass it on to rank r + 1.
    Merge the per-block results with online softmax: keep the output and the logsumexp of the scores so far.
    Backward must work too: either make the P2P exchange differentiable (an autograd.Function whose backward
    sends the grad the other way round the ring) or write the whole backward pass by hand.
    A block from a later rank is entirely in the future of this rank's queries: fully masked, it adds nothing.
    """
    raise NotImplementedError("part 2: ring_attention")


def ulysses_attention(q, k, v, positions, group):
    """Part 4. Ulysses (DeepSpeed-Ulysses): all-to-all from [B, H, S_local, D] to [B, H/n, S, D].

    Each rank then holds every token of its H/n heads and runs ordinary attention on them; a second all-to-all
    brings the output back to [B, H, S_local, D]. Use your all_to_all from mtf/parallel/comm.py (part 3).
    The chunks arrive in rank order, which is position order: the gathered sequence is the whole sequence.
    """
    raise NotImplementedError("part 4: ulysses_attention")
