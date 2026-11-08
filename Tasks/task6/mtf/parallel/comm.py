"""All-to-all built from point-to-point ops (part 3).

    all_to_all_single(input, output_split_sizes, input_split_sizes, group) -> output   plain tensors
    all_to_all(input, output_split_sizes, input_split_sizes, group) -> output          differentiable

Semantics are those of torch.distributed.all_to_all_single: rank r splits `input` along dim 0 into chunks of
input_split_sizes rows and sends chunk j to rank j of the group; the output concatenates, along dim 0, the
chunks received from ranks 0, 1, ..., n-1 (output_split_sizes rows each). Splits may be uneven, and zero.
None means equal splits: input.shape[0] // n rows to and from every rank.
Only P2P ops are allowed here: isend / irecv / batch_isend_irecv (send / recv if you order them carefully).
"""

import torch
import torch.distributed as dist


def all_to_all_single(input: torch.Tensor, output_split_sizes=None, input_split_sizes=None, group=None) -> torch.Tensor:
    """Part 3. All-to-all of the rows of `input` with P2P ops (no autograd)."""
    raise NotImplementedError("part 3: all_to_all_single")


class _AllToAll(torch.autograd.Function):
    """Part 3. Differentiable all-to-all: what is the gradient of a permutation of rows between ranks?"""

    @staticmethod
    def forward(ctx, input, output_split_sizes, input_split_sizes, group):
        raise NotImplementedError("part 3: _AllToAll.forward")

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("part 3: _AllToAll.backward")


def all_to_all(input: torch.Tensor, output_split_sizes=None, input_split_sizes=None, group=None) -> torch.Tensor:
    """Differentiable all-to-all (see the module docstring)."""
    return _AllToAll.apply(input, output_split_sizes, input_split_sizes, group or dist.group.WORLD)
