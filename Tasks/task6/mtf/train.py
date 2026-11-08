"""Training loop. Read it top to bottom: every other module is called from here.

    python -m mtf.train --config configs/debug.toml [--train.steps=100 --train.device=cpu ...]
    torchrun --nproc_per_node=4 -m mtf.train --config configs/debug.toml --train.device=cpu --parallel.dp_shard=4
    torchrun --nproc_per_node=4 -m mtf.train --config configs/debug_cp.toml --parallel.cp=4 --parallel.cp_attn=ring
"""

import dataclasses
import json
import os
import sys
import time
import traceback

import torch
import torch.nn.functional as F
from torch.profiler import record_function

from mtf import checkpoint, dist
from mtf.config import load_config
from mtf.data import build_data
from mtf.metrics import Metrics, peak_flops
from mtf.model import hf
from mtf.model.transformer import Transformer
from mtf.optim import build_optimizer, clip_grad_norm_
from mtf.parallel import parallelize, shard_batch
from mtf.profiling import Profiler


def random_init(args, seed: int) -> dict:
    """Fresh weights as a full state dict, built on CPU (on rank 0): the same on every device and layout."""
    torch.manual_seed(seed)
    model = Transformer(args)
    model.init_weights()
    return model.state_dict()


def main(argv: list[str]) -> list[float]:
    cfg = load_config(argv)
    tc = cfg.train
    rt = dist.init(tc.device, tc.seed, cfg.parallel)
    if rt.is_main:
        os.makedirs(tc.run_dir, exist_ok=True)
        with open(os.path.join(tc.run_dir, "config.json"), "w") as f:
            json.dump(dataclasses.asdict(cfg), f, indent=2)

    # Model: build on "meta" (shapes only, no memory), shard it, materialize it on the device, then fill
    # it from a full state dict on rank 0 (HF weights or a fresh init). Same path for every layout.
    path = hf.resolve(cfg.model.hf_id) if cfg.model.hf_id else ""
    args = hf.args_from_hf(path) if path else cfg.model.args
    with torch.device("meta"):
        model = Transformer(args)
    if rt.mesh is not None:
        parallelize(model, rt.mesh, cfg)  # parameters become DTensors on the mesh
    model.to_empty(device=rt.device)
    if path and cfg.model.load_weights:
        hf.load_hf(model, path)
    else:
        checkpoint.load_full_state_dict(model, random_init(args, tc.seed) if rt.is_main else {})
    n_params = sum(p.numel() for p in model.parameters())

    opt, sched = build_optimizer(model, tc)
    ckpt_dir, start = os.path.join(tc.run_dir, "checkpoints"), 0
    if cfg.ckpt.resume and (ckpt := checkpoint.latest(ckpt_dir)):
        start = checkpoint.load(ckpt, model, opt, sched)
        dist.print0(f"resumed from {ckpt}")
    grad_accum, rem = divmod(tc.global_batch, tc.micro_batch * rt.dp_size)
    assert grad_accum >= 1 and rem == 0, "global_batch must be a multiple of micro_batch * data-parallel size"
    data = build_data(cfg.data, tc.micro_batch, args.vocab_size, cfg.model.hf_id, rt.dp_rank, rt.dp_size, tc.seed)
    for _ in range(start * grad_accum):  # fast-forward the data to where the checkpoint left off
        next(data)
    tokens_per_step = tc.global_batch * cfg.data.seq_len
    peak = peak_flops(rt.device, tc.dtype, tc.peak_tflops) * rt.world_size
    metrics = Metrics(tc.run_dir, rt.device, model.flops_per_token(cfg.data.seq_len), peak, rt.is_main, append=start > 0)
    # bf16: a single process uses autocast; sharded models get bf16 from FSDP's MixedPrecisionPolicy instead
    amp = torch.autocast(rt.device.type, dtype=torch.bfloat16, enabled=tc.dtype == "bf16" and rt.mesh is None)
    layout = dict(zip(rt.mesh.mesh_dim_names, rt.mesh.shape)) if rt.mesh is not None else "single process"
    dist.print0(f"model {cfg.model.hf_id or 'random init'}: {n_params / 1e6:.1f}M params | device {rt.device} | "
                f"dtype {tc.dtype} | {layout} | {grad_accum} micro-batches of {tc.micro_batch}x{cfg.data.seq_len} per rank per step")

    losses = []
    with Profiler(cfg.profile, tc.run_dir, rt.rank, rt.device) as prof:
        for step in range(start + 1, tc.steps + 1):
            t0 = time.perf_counter()
            loss_sum = torch.zeros((), device=rt.device)
            for _ in range(grad_accum):
                batch = shard_batch({k: v.to(rt.device) for k, v in next(data).items()}, rt.mesh)  # CP: my chunk
                with record_function("forward"), amp:
                    logits = model(batch["tokens"], batch["positions"])
                    loss = F.cross_entropy(logits.flatten(0, 1).float(), batch["labels"].flatten()) / grad_accum
                with record_function("backward"):
                    loss.backward()  # sharded: FSDP averages the grads across its ranks
                loss_sum += loss.detach()
            with record_function("optimizer"):
                grad_norm = clip_grad_norm_(model.parameters(), tc.max_norm)
                lr = sched.get_last_lr()[0]
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            prof.step()
            losses.append(dist.mean(loss_sum).item())  # .item() waits for the device, so the step time is real
            echo = step % tc.log_every == 0 or step == tc.steps
            metrics.log(step, losses[-1], lr, dist.item(grad_norm), tokens_per_step, time.perf_counter() - t0, echo)
            if cfg.ckpt.every and step % cfg.ckpt.every == 0:
                checkpoint.save(os.path.join(ckpt_dir, f"step_{step}"), step, model, opt, sched)

    if tc.export_hf:
        hf.export_hf(model, path, tc.export_hf)
        dist.print0(f"exported HF checkpoint to {tc.export_hf}")
    dist.destroy()
    return losses


if __name__ == "__main__":
    # Exit without interpreter shutdown: with HF streaming data, shutdown can abort ("Fatal Python error:
    # PyGILState_Release") or hang forever while still holding GPU memory (seen after a CUDA OOM).
    code = 0
    try:
        main(sys.argv[1:])
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
