"""Our Transformer must match transformers' implementation exactly (fp32, CPU), for every supported family.

Tiny random HF models are created from real config classes, so this runs offline in seconds.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from mtf.model import hf  # noqa: E402
from mtf.model.transformer import Transformer  # noqa: E402

TINY = dict(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
CASES = {
    "mistral": (transformers.MistralConfig, {}),
    "mistral_mha": (transformers.MistralConfig, {"num_key_value_heads": 4}),
    "qwen2": (transformers.Qwen2Config, {}),  # q/k/v biases
    "qwen3": (transformers.Qwen3Config, {"head_dim": 32}),  # qk-norm, head_dim != hidden/heads
    "qwen3_tied": (transformers.Qwen3Config, {"tie_word_embeddings": True}),
}


def make_hf_model(name, path):
    cls, extra = CASES[name]
    torch.manual_seed(0)
    ref = transformers.AutoModelForCausalLM.from_config(cls(**{**TINY, **extra}, rope_theta=500.0)).eval()
    with torch.no_grad():  # perturb everything so biases/norm weights are not trivially 0/1
        for p in ref.parameters():
            p.add_(0.1 * torch.randn_like(p))
    ref.save_pretrained(path)
    return ref


@pytest.mark.parametrize("name", CASES)
def test_logits_match_hf(name, tmp_path):
    ref = make_hf_model(name, tmp_path)
    model = Transformer(hf.args_from_hf(str(tmp_path)))
    hf.load_hf(model, str(tmp_path))
    tokens = torch.randint(0, TINY["vocab_size"], (2, 24))
    with torch.no_grad():
        torch.testing.assert_close(model(tokens), ref(tokens).logits, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name", ["qwen3", "qwen3_tied"])
def test_export_roundtrip(name, tmp_path):
    make_hf_model(name, tmp_path / "src")
    model = Transformer(hf.args_from_hf(str(tmp_path / "src")))
    hf.load_hf(model, str(tmp_path / "src"))
    with torch.no_grad():  # "train" a bit so we export something different from the source
        model.layers["0"].self_attn.o_proj.weight.mul_(1.5)
    hf.export_hf(model, str(tmp_path / "src"), str(tmp_path / "out"), dtype=torch.float32)
    back = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "out", dtype=torch.float32).eval()
    tokens = torch.randint(0, TINY["vocab_size"], (2, 24))
    with torch.no_grad():
        torch.testing.assert_close(model(tokens), back(tokens).logits, rtol=1e-5, atol=1e-5)


def test_positions_drive_rope():
    # forward(tokens) == forward(tokens, explicit arange positions); shifted positions change the output
    model = Transformer(hf.ModelArgs())
    model.init_weights()
    tokens = torch.randint(0, 256, (2, 16))
    pos = torch.arange(16).expand(2, -1)
    with torch.no_grad():
        torch.testing.assert_close(model(tokens), model(tokens, pos))
        assert not torch.allclose(model(tokens), model(tokens, pos + 5))
