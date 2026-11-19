"""Metrics: one line per step to stdout + one JSON object per step to <run_dir>/metrics.jsonl (rank 0 only)."""

import json
import os
import resource
import sys

import torch

# Dense bf16 tensor-core peak FLOP/s from vendor spec sheets, for MFU. Add your GPU here.
PEAK_BF16_FLOPS = {"H100": 989e12, "H200": 989e12, "A100": 312e12, "B200": 2250e12}


def peak_flops(device: torch.device, dtype: str, override_tflops: float = 0.0) -> float:
    if override_tflops:
        return override_tflops * 1e12
    if device.type != "cuda" or dtype != "bf16":
        return 0.0  # MFU is reported only for bf16 on known GPUs
    name = torch.cuda.get_device_name(device)
    return next((v for k, v in PEAK_BF16_FLOPS.items() if k in name), 0.0)


def memory_gb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e9  # peak since start
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9  # current, incl. allocator cache
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # peak process RSS
    return rss / 1e9 if sys.platform == "darwin" else rss / 1e6  # bytes on macOS, KB on Linux


class Metrics:
    def __init__(self, run_dir: str, device: torch.device, flops_per_token: float, peak: float, enabled: bool = True, append: bool = False):
        """peak: FLOP/s of all devices together (0: no MFU). append: keep earlier rows (resumed run)."""
        self.device, self.flops_per_token, self.peak, self.enabled = device, flops_per_token, peak, enabled
        if enabled:
            os.makedirs(run_dir, exist_ok=True)
            self.file = open(os.path.join(run_dir, "metrics.jsonl"), "a" if append else "w")

    def log(self, step: int, loss: float, lr: float, grad_norm: float, tokens: int, seconds: float, echo: bool = True):
        """Every call goes to metrics.jsonl; `echo` controls the stdout line (train.log_every)."""
        if not self.enabled:
            return
        row = {
            "step": step,
            "loss": loss,
            "lr": lr,
            "grad_norm": grad_norm,
            "tok_per_s": tokens / seconds,
            "step_s": seconds,
            "mem_gb": memory_gb(self.device),
        }
        if self.peak:
            row["mfu"] = self.flops_per_token * row["tok_per_s"] / self.peak
        self.file.write(json.dumps(row) + "\n")
        self.file.flush()
        if not echo:
            return
        mfu = f" | mfu {row['mfu']:.1%}" if self.peak else ""
        print(
            f"step {step:>5} | loss {loss:.4f} | lr {lr:.2e} | grad_norm {grad_norm:.3f} | "
            f"tok/s {row['tok_per_s']:,.0f}{mfu} | mem {row['mem_gb']:.2f}GB | {seconds:.2f}s",
            flush=True,
        )
