# AGENTS.md

`mtf` is a minimal, DTensor-native LLM training framework, cut down for a homework on FSDP: data parallelism
with torch's FSDP2 (`fully_shard`) is given; a hand-written FSDP in `mtf/parallel/my_fsdp.py` is the task
(README.md).

## Commands

```bash
uv sync                                                          # torch version is pinned in uv.lock
uv run pytest -q                                                 # CPU only, ≈4 min
uv run pytest -q -m part2                                        # one part of the homework
uv run python tools/grade.py                                     # tests -> points, ≈3 min
uv run python -m mtf.train --config configs/debug.toml           # tiny random model, synthetic data, seconds
uv run torchrun --master_addr=127.0.0.1 --master_port=29511 --nproc_per_node=4 -m mtf.train --config configs/debug.toml --train.device=cpu --parallel.fsdp_impl=mine
uv run python tools/compare.py runs/<run> reference/tinystories_fsdp_1proc.jsonl
```

Any config key can be overridden: `--train.lr=1e-3 --train.device=cpu --model.args.n_layers=2`.

## Where things live (one concept per file)

| File | Responsibility |
|---|---|
| `mtf/train.py` | the training loop; read this first |
| `mtf/config.py` | all knobs as dataclasses; TOML + dotted CLI overrides |
| `mtf/dist.py` | device/backend selection, process group init, device mesh (dp_replicate, dp_shard) |
| `mtf/parallel/__init__.py` | `parallelize()`: picks the FSDP implementation (`parallel.fsdp_impl`) |
| `mtf/parallel/fsdp.py` | DDP / FSDP / HSDP via torch's `fully_shard` (FSDP2) |
| `mtf/parallel/my_fsdp.py` | hand-written FSDP: sharding, all-gather / reduce-scatter hooks, prefetch |
| `mtf/checkpoint.py` | DCP checkpoints; full state dict <-> any layout |
| `mtf/model/transformer.py` | the model (Qwen2 / Qwen3 / Mistral via flags) |
| `mtf/model/hf.py` | HF config -> ModelArgs, load weights, export to HF format |
| `mtf/data.py` | synthetic, HF streaming and pre-tokenized file data; per-dp-rank sharding |
| `mtf/optim.py` | AdamW + warmup/cosine schedule, grad clipping |
| `mtf/metrics.py` | stdout + `metrics.jsonl`, tok/s, memory |
| `mtf/profiling.py` | torch.profiler traces |
| `tests/dist_checks.py` | distributed checks of `my_fsdp.py`, run under torchrun by `tests/test_my_fsdp.py` |

## Invariants (keep these when changing code)

- Only `mtf/dist.py` knows about devices, backends and process groups.
- Module names in `transformer.py` match HF minus the `model.` prefix (HF load/export rely on it).
- Model API: `forward(tokens, positions)`; `layers` is a `ModuleDict` keyed `"0".."L-1"`: both FSDP
  implementations make one unit per block plus the root unit.
- `my_fsdp.py` keeps its API (`FSDPParam.sharded/unsharded`, `FSDPUnit` methods, `module.fsdp_unit`,
  `Context.units`): `tests/dist_checks.py` instruments it.
- A new knob is a new dataclass field with a non-None default in `config.py`; nothing else to register.
- `python -m mtf.train` ends with `os._exit` (HF streaming can hang interpreter shutdown): do cleanup inside
  `main()`, not in `atexit` handlers or destructors.
- A single process trains a plain module (no mesh): it is the reference. `parallelize()` runs only under
  torchrun with more than one process, before weights exist (meta device).
- Synthetic and file data are independent of the data-parallel size (each rank slices one global micro-batch).
  `tests/test_parallel.py` relies on it.
- Correctness is checked by equivalence: model vs HF logits (`tests/test_model_parity.py`); every parallel
  layout vs a single process, loss and grad norm per step, and `fsdp_impl=mine` vs FSDP2
  (`tests/test_parallel.py`, CPU/gloo).
