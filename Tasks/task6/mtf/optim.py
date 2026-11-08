"""Optimizer + LR schedule: AdamW (no weight decay on 1-D params: norms, biases), linear warmup, cosine decay.
Plus grad clipping that works whatever the parallel layout.

Works unchanged on DTensor parameters (FSDP), so parallelism never touches this file.
"""

import math

import torch

from mtf.config import TrainCfg
from mtf.dist import full


def build_optimizer(model, cfg: TrainCfg):
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {"params": [p for p in params if p.ndim >= 2], "weight_decay": cfg.weight_decay},
        {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=1e-8)

    def lr_factor(step):  # step counts from 0
        if step < cfg.warmup:
            return (step + 1) / cfg.warmup
        progress = min(1.0, (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup))
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)


def clip_grad_norm_(params, max_norm: float) -> torch.Tensor:
    """Scale grads so their global L2 norm is at most max_norm; returns the norm before clipping.

    Like torch.nn.utils.clip_grad_norm_, but parameters may be DTensors on different meshes:
    norms are computed per mesh, then combined. Sums of squares are accumulated in fp64: in fp32, the norm of a
    10M-element grad (an embedding) is off by ~1e-4, and by a different amount for each sharding, so layouts
    would clip differently and their training curves drift apart.
    """
    grads = [p.grad for p in params if p.grad is not None]
    by_mesh = {}
    for g in grads:
        by_mesh.setdefault(getattr(g, "device_mesh", None), []).append(g)
    norms = [torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g, dtype=torch.float64) for g in gs])) for gs in by_mesh.values()]
    total = torch.linalg.vector_norm(torch.stack([full(n) for n in norms]))
    scale = min(1.0, max_norm / (total.item() + 1e-6))
    for g in grads:
        g.mul_(scale)
    return total
