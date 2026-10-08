"""THE key correctness test. Hugging Face's Llama is used here only as a referee:
my weights go into a random-init HF model; both must produce the same logits."""
import pytest
import torch
from transformers import LlamaForCausalLM

from model.config import ModelConfig
from model.convert_hf import to_hf_config, to_hf_state_dict
from model.model import TinySQL
from tests.test_attention_backends import small_cfg


def build_pair(cfg: ModelConfig):
    mine = TinySQL(cfg).eval()
    hf = LlamaForCausalLM(to_hf_config(cfg), ).eval()
    hf.config._attn_implementation = "eager"
    missing = hf.load_state_dict(to_hf_state_dict(mine.state_dict(), cfg.n_layers), strict=False)
    assert not missing.unexpected_keys
    assert all("rotary" in k for k in missing.missing_keys) or not missing.missing_keys or cfg.tie_embeddings
    return mine, hf


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("tie", [True, False])
def test_small_model_logits_match_hf(fused, tie):
    cfg = small_cfg(fused_attention=fused, tie_embeddings=tie)
    mine, hf = build_pair(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 17))
    with torch.no_grad():
        mine_logits, _, _ = mine(ids)
        hf_logits = hf(ids).logits
    assert (mine_logits - hf_logits).abs().max() < 1e-4


def test_full_size_model_logits_match_hf():
    cfg = ModelConfig()
    mine, hf = build_pair(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 24))
    with torch.no_grad():
        mine_logits, _, _ = mine(ids)
        hf_logits = hf(ids).logits
    assert (mine_logits - hf_logits).abs().max() < 1e-4
