"""Tokenize the start of a HF text dataset once into a flat uint16 token file (data.source = "file").

    uv run python tools/pretokenize.py --out data/tinystories.bin --tokens 2000000

Documents are joined with EOS, like data.source = "hf". uint16 holds ids < 65536 (SmolLM2's tokenizer: 49152).
"""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--tokens", type=int, default=2_000_000)
p.add_argument("--dataset", default="roneneldan/TinyStories")
p.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(a.tokenizer)
assert len(tok) <= 65536, "uint16 token ids"
ids = []
for example in load_dataset(a.dataset, split="train", streaming=True):
    ids.extend(tok(example["text"])["input_ids"] + [tok.eos_token_id])
    if len(ids) >= a.tokens:
        break
torch.tensor(ids[: a.tokens], dtype=torch.int32).to(torch.uint16).numpy().tofile(a.out)
print(f"{a.out}: {a.tokens:,} tokens, vocab {len(tok)}")
