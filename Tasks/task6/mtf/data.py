"""Data: an endless iterator of {"tokens", "labels", "positions"} batches, each [micro_batch, seq_len].

synthetic: a fixed pool of random sequences -- memorizable, so loss goes down; no downloads. Every rank
           draws the same global micro-batch and keeps its slice, so the data is identical whatever the
           data-parallel size: that is what lets tests compare parallel layouts with a single process.
hf:        HF `datasets` streaming -> tokenizer -> documents joined with EOS -> packed into seq_len+1 windows.
           Each data-parallel rank streams a disjoint shard of the dataset.
file:      the same, pre-tokenized once by tools/pretokenize.py: consecutive seq_len+1 windows of a uint16
           token file, read like synthetic (identical whatever the data-parallel size), no downloads.
Context parallel ranks of one dp group get the same batch: each keeps its part (parallel.shard_batch).
Replace this file to change data.
"""

import itertools
import os

import torch

from mtf.config import DataCfg


def build_data(cfg: DataCfg, micro_batch: int, vocab_size: int, tokenizer: str = "", dp_rank: int = 0, dp_size: int = 1, seed: int = 0):
    if cfg.source == "synthetic":
        windows = _synthetic(cfg, micro_batch, vocab_size, seed, dp_rank, dp_size)
    elif cfg.source == "hf":
        windows = _hf(cfg, micro_batch, vocab_size, cfg.tokenizer or tokenizer, dp_rank, dp_size)
    elif cfg.source == "file":
        windows = _file(cfg, micro_batch, vocab_size, dp_rank, dp_size)
    else:
        raise ValueError(f"unknown data source {cfg.source!r}")
    for w in windows:  # w: [micro_batch, seq_len + 1]
        positions = torch.arange(cfg.seq_len).expand(w.shape[0], -1)
        yield {"tokens": w[:, :-1], "labels": w[:, 1:], "positions": positions}


def _synthetic(cfg: DataCfg, batch: int, vocab_size: int, seed: int, dp_rank: int, dp_size: int):
    g = torch.Generator().manual_seed(seed)  # same seed on every rank
    pool = torch.randint(0, vocab_size, (cfg.synthetic_pool, cfg.seq_len + 1), generator=g)
    while True:
        idx = torch.randint(0, cfg.synthetic_pool, (batch * dp_size,), generator=g)  # the global micro-batch
        yield pool[idx[dp_rank * batch : (dp_rank + 1) * batch]]


def _file(cfg: DataCfg, batch: int, vocab_size: int, dp_rank: int, dp_size: int):
    tokens = torch.from_file(cfg.path, dtype=torch.uint16, size=os.path.getsize(cfg.path) // 2).long()  # uint16: few ops
    windows = tokens[: len(tokens) // (cfg.seq_len + 1) * (cfg.seq_len + 1)].view(-1, cfg.seq_len + 1)
    if int(windows.max()) >= vocab_size:
        raise ValueError(f"{cfg.path} has token ids >= vocab_size={vocab_size}")
    step = batch * dp_size  # one global micro-batch per step; this rank keeps its slice
    for i in itertools.count():
        start = i * step % (len(windows) - step + 1)
        yield windows[start + dp_rank * batch : start + (dp_rank + 1) * batch]


def _hf(cfg: DataCfg, batch: int, vocab_size: int, tokenizer: str, dp_rank: int, dp_size: int):
    from datasets import load_dataset
    from datasets.distributed import split_dataset_by_node
    from transformers import AutoTokenizer

    if not tokenizer:
        raise ValueError("data.source=hf needs data.tokenizer (or model.hf_id)")
    tok = AutoTokenizer.from_pretrained(tokenizer)
    if len(tok) > vocab_size:
        raise ValueError(f"tokenizer has {len(tok)} tokens but the model only {vocab_size}")
    ds = load_dataset(cfg.dataset, cfg.subset or None, split=cfg.split, streaming=True)
    ds = split_dataset_by_node(ds, rank=dp_rank, world_size=dp_size)
    eos = [tok.eos_token_id] if tok.eos_token_id is not None else []
    buf, rows, epoch = [], [], 0
    while True:
        ds.set_epoch(epoch)
        for example in ds:
            buf.extend(tok(example[cfg.text_field])["input_ids"] + eos)
            while len(buf) > cfg.seq_len:
                rows.append(buf[: cfg.seq_len + 1])
                buf = buf[cfg.seq_len + 1 :]
                if len(rows) == batch:
                    yield torch.tensor(rows)
                    rows = []
        epoch += 1
