"""Profiling: torch.profiler timeline + memory, per rank. Call prof.step() once per training step.

Outputs in <run_dir>/profile/:
  trace_rank{R}.json      timeline: open in https://ui.perfetto.dev
  ops_rank{R}.txt         top ops by self time
  memory_rank{R}.pickle   CUDA only: allocator history, drag into https://pytorch.org/memory_viz
CPU and MPS record CPU-side activity (with allocations if profile.memory); on MPS, GPU kernels run
asynchronously, so CPU time shows dispatch, not kernel time.
"""

import os

import torch
from torch.profiler import ProfilerActivity, profile, schedule

from mtf.config import ProfileCfg


class Profiler:
    def __init__(self, cfg: ProfileCfg, run_dir: str, rank: int, device: torch.device):
        self.cfg, self.rank, self.cuda = cfg, rank, device.type == "cuda"
        self.out = os.path.join(run_dir, "profile")
        self.prof, self.steps = None, 0

    def __enter__(self):
        if not self.cfg.enabled:
            return self
        os.makedirs(self.out, exist_ok=True)
        c = self.cfg
        self.prof = profile(
            activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if self.cuda else []),
            schedule=schedule(wait=c.wait, warmup=c.warmup, active=c.active, repeat=1),
            on_trace_ready=self._save,
            record_shapes=True,
            profile_memory=c.memory,
        )
        self.prof.__enter__()
        if self.cuda and c.memory:
            torch.cuda.memory._record_memory_history(max_entries=100_000)
        return self

    def step(self):
        if self.prof is None:
            return
        self.prof.step()
        self.steps += 1
        if self.cuda and self.cfg.memory and self.steps == self.cfg.wait + self.cfg.warmup + self.cfg.active:
            torch.cuda.memory._dump_snapshot(os.path.join(self.out, f"memory_rank{self.rank}.pickle"))
            torch.cuda.memory._record_memory_history(enabled=None)

    def _save(self, prof):
        prof.export_chrome_trace(os.path.join(self.out, f"trace_rank{self.rank}.json"))
        sort = "self_cuda_time_total" if self.cuda else "self_cpu_time_total"
        with open(os.path.join(self.out, f"ops_rank{self.rank}.txt"), "w") as f:
            f.write(prof.key_averages().table(sort_by=sort, row_limit=40))
        if self.rank == 0:
            print(f"profiler: wrote {self.out}/trace_rank{self.rank}.json", flush=True)

    def __exit__(self, *exc):
        if self.prof is not None:
            self.prof.__exit__(*exc)
