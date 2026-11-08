"""Compare a run with the reference, step by step: loss and grad norm.

    uv run python tools/compare.py runs/ts_cp4_ring reference/tinystories_1proc.jsonl

fp32 runs that differ only in summation order (another layout, another CPU) agree almost exactly at first and
drift apart later: training amplifies rounding. So the first 40 steps are the strict check.
"""

import json
import sys
from pathlib import Path

STRICT = 40


def rows(path: str) -> dict[int, dict]:
    p = Path(path)
    p = p / "metrics.jsonl" if p.is_dir() else p
    return {r["step"]: r for r in map(json.loads, open(p))}


run, ref = rows(sys.argv[1]), rows(sys.argv[2])
steps = sorted(run.keys() & ref.keys())
if not steps:
    raise SystemExit("no common steps")
diff = {s: {k: abs(run[s][k] - ref[s][k]) / abs(ref[s][k]) for k in ("loss", "grad_norm")} for s in steps}
print(f"{'step':>5} {'loss ref':>10} {'loss':>10} {'rel diff':>9} {'grad_norm rel diff':>19}")
for s in steps:
    if s == steps[0] or s % 10 == 0 or s == steps[-1]:
        print(f"{s:>5} {ref[s]['loss']:>10.4f} {run[s]['loss']:>10.4f} {diff[s]['loss']:>9.1e} {diff[s]['grad_norm']:>19.1e}")
for name, chosen in ((f"steps 1-{STRICT}", [s for s in steps if s <= STRICT]), (f"all {len(steps)} steps", steps)):
    if chosen:
        worst = {k: max(diff[s][k] for s in chosen) for k in ("loss", "grad_norm")}
        print(f"{name}: max relative difference: loss {worst['loss']:.1e}, grad_norm {worst['grad_norm']:.1e}")
