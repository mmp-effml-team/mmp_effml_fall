"""Config: nested dataclasses <- TOML file <- dotted CLI overrides.

    python -m mtf.train --config configs/debug.toml --train.lr=1e-3 --model.args.n_layers=2

To add a knob, add a field (with a non-None default) to a dataclass below. Nothing else to register.
"""

import dataclasses
import tomllib
from dataclasses import dataclass, field

from mtf.model.transformer import ModelArgs


@dataclass
class ModelCfg:
    hf_id: str = ""  # HF repo id or local dir. Empty: random init from `args`
    load_weights: bool = True  # with hf_id: False = HF architecture, random init
    args: ModelArgs = field(default_factory=ModelArgs)  # used only when hf_id is empty


@dataclass
class DataCfg:
    source: str = "synthetic"  # synthetic | hf | file
    seq_len: int = 256
    path: str = ""  # file: pre-tokenized uint16 tokens (tools/pretokenize.py)
    synthetic_pool: int = 64  # synthetic: number of distinct sequences (memorizable -> loss goes down)
    dataset: str = "roneneldan/TinyStories"  # hf: dataset id
    subset: str = ""  # hf: dataset config name
    split: str = "train"
    text_field: str = "text"
    tokenizer: str = ""  # default: model.hf_id


@dataclass
class TrainCfg:
    steps: int = 20
    micro_batch: int = 4  # sequences per forward pass on each rank
    global_batch: int = 4  # sequences per optimizer step, all ranks; must be a multiple of micro_batch
    lr: float = 3e-4
    min_lr_ratio: float = 0.1  # cosine decays to lr * min_lr_ratio
    warmup: int = 5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_norm: float = 1.0  # grad clipping
    dtype: str = "fp32"  # fp32 | bf16 (autocast)
    peak_tflops: float = 0.0  # MFU denominator; 0: look up the GPU in metrics.PEAK_BF16_FLOPS
    device: str = "auto"  # auto | cuda | cpu | mps
    seed: int = 0
    log_every: int = 1
    run_dir: str = "runs/default"
    export_hf: str = ""  # if set: save the trained model in HF format to this dir at the end


@dataclass
class ParallelCfg:  # degrees of the device mesh (dp_replicate, dp_shard, cp); product = world size
    dp_replicate: int = 1  # replicas: DDP when dp_shard=1, HSDP when both > 1
    dp_shard: int = -1  # FSDP shards; -1: whatever is left of the world size
    cp: int = 1  # context parallel: each rank holds 1/cp of every sequence
    cp_attn: str = "allgather"  # how attention sees the whole sequence: allgather | ring | ulysses


@dataclass
class CkptCfg:
    every: int = 0  # save every N steps to <run_dir>/checkpoints/step_N (0: never)
    resume: bool = False  # start from the latest checkpoint in <run_dir>/checkpoints, if there is one


@dataclass
class ProfileCfg:
    enabled: bool = False
    wait: int = 3  # steps to skip, then `warmup` steps traced but discarded, then `active` steps recorded
    warmup: int = 1
    active: int = 2
    memory: bool = True  # CPU: allocations in the trace; CUDA: also a memory snapshot for pytorch.org/memory_viz


@dataclass
class Config:
    model: ModelCfg = field(default_factory=ModelCfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    parallel: ParallelCfg = field(default_factory=ParallelCfg)
    ckpt: CkptCfg = field(default_factory=CkptCfg)
    profile: ProfileCfg = field(default_factory=ProfileCfg)


def _update(obj, values: dict, prefix=""):
    for key, value in values.items():
        if not hasattr(obj, key):
            raise ValueError(f"unknown config key: {prefix}{key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            _update(current, value, f"{prefix}{key}.")
        else:
            setattr(obj, key, type(current)(value) if type(current) is float else value)


def _parse(text: str, like):
    if isinstance(like, bool):
        if text.lower() not in ("true", "false", "1", "0"):
            raise ValueError(f"expected a bool, got {text!r}")
        return text.lower() in ("true", "1")
    return type(like)(text)


def load_config(argv: list[str]) -> Config:
    cfg = Config()
    overrides = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--config":
            arg, i = f"--config={argv[i + 1]}", i + 1
        if not arg.startswith("--") or "=" not in arg:
            raise ValueError(f"expected --key=value, got {arg!r}")
        key, value = arg[2:].split("=", 1)
        if key == "config":
            with open(value, "rb") as f:
                _update(cfg, tomllib.load(f))
        else:
            overrides.append((key, value))
        i += 1
    for key, value in overrides:  # CLI wins over the file, whatever the order
        *path, name = key.split(".")
        obj = cfg
        for p in path:
            obj = getattr(obj, p)
        if not hasattr(obj, name) or dataclasses.is_dataclass(getattr(obj, name)):
            raise ValueError(f"unknown config key: {key}")
        setattr(obj, name, _parse(value, getattr(obj, name)))
    return cfg
