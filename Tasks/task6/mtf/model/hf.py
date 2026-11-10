"""Hugging Face in/out. Our module names == HF names minus "model.", so loading needs no key mapping.

    path = resolve("Qwen/Qwen3-0.6B")               # download once (rank 0), or pass a local dir
    model = Transformer(args_from_hf(path)); load_hf(model, path)
    export_hf(model, path, "out/")                  # safetensors + config/tokenizer copied from `path`

load_hf works for any parallel layout: rank 0 reads the files, set_model_state_dict scatters into DTensors.
"""

import glob
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

from mtf.checkpoint import get_full_state_dict, load_full_state_dict
from mtf.dist import barrier, is_main
from mtf.model.transformer import ModelArgs

SUPPORTED = {"mistral", "qwen2", "qwen3"}
FILES = ["*.json", "*.safetensors", "*.txt", "*.model"]  # config, weights, tokenizer


def resolve(path_or_id: str) -> str:
    if os.path.isdir(path_or_id):
        return path_or_id
    from huggingface_hub import snapshot_download

    if is_main():  # one download; the other ranks then hit the local cache
        snapshot_download(path_or_id, allow_patterns=FILES)
    barrier()
    return snapshot_download(path_or_id, allow_patterns=FILES)


def args_from_hf(path: str) -> ModelArgs:
    with open(os.path.join(path, "config.json")) as f:
        c = json.load(f)
    kind = c["model_type"]
    if kind not in SUPPORTED:
        raise NotImplementedError(f"model_type={kind!r}; supported: {sorted(SUPPORTED)}")
    rope = c.get("rope_parameters") or c.get("rope_scaling") or {}  # transformers v5 | v4 config format
    if rope.get("rope_type", rope.get("type", "default")) != "default":
        raise NotImplementedError(f"rope scaling {rope} (only plain RoPE for now)")
    if c.get("mlp_bias") or (c.get("attention_bias") and kind != "qwen2"):
        raise NotImplementedError("attention_bias / mlp_bias")
    # Mistral's sliding_window is ignored: exact as long as seq_len <= sliding_window.
    return ModelArgs(
        vocab_size=c["vocab_size"],
        dim=c["hidden_size"],
        n_layers=c["num_hidden_layers"],
        n_heads=c["num_attention_heads"],
        n_kv_heads=c.get("num_key_value_heads") or c["num_attention_heads"],
        head_dim=c.get("head_dim") or 0,
        ffn_dim=c["intermediate_size"],
        norm_eps=c["rms_norm_eps"],
        rope_theta=c.get("rope_theta") or rope.get("rope_theta", 10000.0),
        qkv_bias=kind == "qwen2",
        qk_norm=kind == "qwen3",
        tie_embeddings=c.get("tie_word_embeddings", False),
    )



def load_hf(model, path: str):
    """Copy HF weights into an already materialized model (after to_empty / parallelize)."""
    sd = {}
    if is_main():
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            sd.update(load_file(f))
        sd = {k.removeprefix("model."): v for k, v in sd.items()}
        if model.lm_head is None:
            sd.pop("lm_head.weight", None)  # tied: some checkpoints still store a copy
    load_full_state_dict(model, sd)  # strict: a missing/unexpected key means an unsupported architecture feature


def export_hf(model, src_path: str, out_dir: str, dtype=torch.bfloat16):
    """Write HF-loadable safetensors; config + tokenizer files are copied from the source checkpoint."""
    sd = get_full_state_dict(model)  # a collective: every rank takes part, rank 0 gets the result
    if is_main():
        os.makedirs(out_dir, exist_ok=True)
        tensors = {(k if k == "lm_head.weight" else "model." + k): v.detach().to("cpu", dtype).contiguous() for k, v in sd.items()}
        save_file(tensors, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
        for f in glob.glob(os.path.join(src_path, "*")):
            name = os.path.basename(f)
            if not name.endswith((".safetensors", ".index.json")) and os.path.isfile(f):
                shutil.copy(f, out_dir)
    barrier()
