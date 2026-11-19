"""Moving state in and out of a model, whatever its parallel layout.

Training checkpoints use torch.distributed.checkpoint (DCP): every rank writes its own shards, and loading
reshards, so a run saved with FSDP on 4 ranks resumes with DDP on 2 ranks or in a single process.
  <run_dir>/checkpoints/step_N/   model + AdamW state + LR schedule + step
Full state dicts (HF loading/export, fresh init) are built on rank 0 and scattered into the shards.
"""

import os
import re

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)


def load_full_state_dict(model, sd: dict):
    """sd: the full (unsharded) state dict on rank 0, {} on the other ranks."""
    if dist.is_initialized():
        set_model_state_dict(model, sd, options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))
    else:
        model.load_state_dict(sd)  # strict: a missing/unexpected key is an error


def get_full_state_dict(model) -> dict:
    """The full state dict on CPU, on rank 0 (other ranks get {})."""
    if dist.is_initialized():
        return get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def save(path: str, step: int, model, opt, sched):
    model_sd, optim_sd = get_state_dict(model, opt)
    dcp.save({"model": model_sd, "optim": optim_sd, "sched": sched.state_dict(), "step": step}, checkpoint_id=path)


def load(path: str, model, opt, sched) -> int:
    """Load a checkpoint in place; returns the step it was saved at."""
    model_sd, optim_sd = get_state_dict(model, opt)  # the target layout; DCP fills it from whatever was saved
    state = {"model": model_sd, "optim": optim_sd, "sched": sched.state_dict(), "step": 0}
    dcp.load(state, checkpoint_id=path)
    set_state_dict(model, opt, model_state_dict=state["model"], optim_state_dict=state["optim"])
    sched.load_state_dict(state["sched"])
    return state["step"]


def latest(ckpt_dir: str) -> str:
    """Path of the newest step_N checkpoint in ckpt_dir, or "" if there is none."""
    steps = [int(m.group(1)) for d in os.listdir(ckpt_dir) if (m := re.fullmatch(r"step_(\d+)", d))] if os.path.isdir(ckpt_dir) else []
    return os.path.join(ckpt_dir, f"step_{max(steps)}") if steps else ""
