import torch
import torch.nn.functional as F

from model.attention import GroupedQueryAttention
from model.block import DecoderBlock
from model.config import ModelConfig
from model.mlp import SwiGLU, silu
from model.rope import RotaryEmbedding
from tests.test_attention_backends import small_cfg


def test_own_silu_matches_torch():
    x = torch.randn(1000) * 10
    assert torch.allclose(silu(x), F.silu(x), atol=1e-6)


def test_swiglu_matches_formula_and_param_count():
    cfg = small_cfg()
    mlp = SwiGLU(cfg)
    x = torch.randn(2, 5, cfg.d_model)
    expected = (F.silu(x @ mlp.gate_proj.weight.T) * (x @ mlp.up_proj.weight.T)) @ mlp.down_proj.weight.T
    assert torch.allclose(mlp(x), expected, atol=1e-6)
    assert sum(p.numel() for p in SwiGLU(ModelConfig()).parameters()) == 2_654_208


def test_block_param_count():
    assert sum(p.numel() for p in DecoderBlock(ModelConfig()).parameters()) == 3_540_096


def test_block_is_identity_when_branches_are_zero():
    # with zeroed output projections both branches add 0, so the residual stream passes through unchanged
    cfg = small_cfg()
    block = DecoderBlock(cfg)
    rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len)
    block.attn.o_proj.weight.data.zero_()
    block.mlp.down_proj.weight.data.zero_()
    x = torch.randn(2, 6, cfg.d_model)
    y, _ = block(x, rope)
    assert torch.equal(y, x)


def test_fused_and_manual_attention_agree_incl_cache():
    cfg_m, cfg_f = small_cfg(fused_attention=False), small_cfg(fused_attention=True)
    manual, fused = GroupedQueryAttention(cfg_m), GroupedQueryAttention(cfg_f)
    fused.load_state_dict(manual.state_dict())
    rope = RotaryEmbedding(cfg_m.head_dim, cfg_m.max_seq_len)
    x = torch.randn(2, 10, cfg_m.d_model)
    ym, cm = manual(x[:, :6], rope)
    yf, cf = fused(x[:, :6], rope)
    assert torch.allclose(ym, yf, atol=1e-5)
    ym2, _ = manual(x[:, 6:7], rope, start_pos=6, past_kv=cm)     # single-token decode path
    yf2, _ = fused(x[:, 6:7], rope, start_pos=6, past_kv=cf)
    assert torch.allclose(ym2, yf2, atol=1e-5)


def test_block_kv_cache_decode_matches_full_forward():
    cfg = small_cfg()
    block = DecoderBlock(cfg)
    rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len)
    x = torch.randn(1, 9, cfg.d_model)
    full, _ = block(x, rope)
    out, cache = block(x[:, :5], rope)
    pieces = [out]
    for t in range(5, 9):
        out, cache = block(x[:, t : t + 1], rope, start_pos=t, past_kv=cache)
        pieces.append(out)
    assert torch.allclose(full, torch.cat(pieces, dim=1), atol=1e-5)
