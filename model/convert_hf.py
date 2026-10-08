"""Weight-name mapping between TinySQL and Hugging Face's Llama. Used ONLY by the oracle test:
my weights are copied into a random-initialised HF model to prove my architecture is correct."""
from transformers import LlamaConfig

from model.config import ModelConfig


def to_hf_config(cfg: ModelConfig) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.d_model, intermediate_size=cfg.d_ff,
        num_hidden_layers=cfg.n_layers, num_attention_heads=cfg.n_heads, num_key_value_heads=cfg.n_kv_heads,
        max_position_embeddings=cfg.max_seq_len, rms_norm_eps=cfg.norm_eps,
        rope_parameters={"rope_type": "default", "rope_theta": cfg.rope_theta},
        tie_word_embeddings=cfg.tie_embeddings, attention_bias=False, mlp_bias=False,
    )


def to_hf_state_dict(mine: dict, n_layers: int) -> dict:
    out = {"model.embed_tokens.weight": mine["embed.weight"], "model.norm.weight": mine["final_norm.weight"]}
    for i in range(n_layers):
        b, h = f"blocks.{i}.", f"model.layers.{i}."
        for p in ("q", "k", "v", "o"):
            out[f"{h}self_attn.{p}_proj.weight"] = mine[f"{b}attn.{p}_proj.weight"]
        for p in ("gate", "up", "down"):
            out[f"{h}mlp.{p}_proj.weight"] = mine[f"{b}mlp.{p}_proj.weight"]
        out[f"{h}input_layernorm.weight"] = mine[f"{b}attn_norm.weight"]
        out[f"{h}post_attention_layernorm.weight"] = mine[f"{b}mlp_norm.weight"]
    if "lm_head.weight" in mine:
        out["lm_head.weight"] = mine["lm_head.weight"]
    return out
