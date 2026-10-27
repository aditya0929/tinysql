"""Export a TinySQL checkpoint to a standard Hugging Face folder (distribution only; training never uses this).

    python -m scripts.export_hf --ckpt checkpoints/tinysql_125m_base.pt --out release/TinySQL-125M-base
    python -m scripts.export_hf --ckpt checkpoints/tinysql_125m_wikisql_lr5e-05_s0.pt --out release/TinySQL-125M-WikiSQL --sql
"""
import argparse
import json
from pathlib import Path

import torch

from model.config import ModelConfig
from model.convert_hf import to_hf_config, to_hf_state_dict
from tokenizer.bpe import BPETokenizer

TOKENIZER = "tokenizer/tinysql_bpe.json"


def bytes_to_unicode():
    """GPT-2's reversible map from the 256 byte values to printable characters (what `tokenizers` ByteLevel uses)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def to_hf_tokenizer(tok: BPETokenizer):
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers
    b2u = bytes_to_unicode()
    s = lambda bs: "".join(b2u[b] for b in bs)
    vocab = {s(bs): i for i, bs in tok.vocab.items()}
    merges = [(s(tok.vocab[a]), s(tok.vocab[b])) for a, b in tok.merges]
    hf = Tokenizer(models.BPE(vocab=vocab, merges=merges, fuse_unk=False, byte_fallback=False))
    hf.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(tok.pattern), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    hf.decoder = decoders.ByteLevel()
    hf.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in tok.special_tokens])
    for t in tok.special_tokens:
        assert hf.token_to_id(t) == tok.special_to_id[t], t
    return hf


def load_model_state(ckpt: str):
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**c["cfg"]["model"]) if "cfg" in c else ModelConfig()
    return cfg, c["model"]


def export(ckpt: str, out: str, sql: bool):
    from transformers import GenerationConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg, state = load_model_state(ckpt)
    hf_cfg = to_hf_config(cfg)
    tok = BPETokenizer.load(TOKENIZER)
    eos, pad = tok.special_to_id["<|endoftext|>"], tok.special_to_id["<|pad|>"]
    hf_cfg.bos_token_id, hf_cfg.eos_token_id, hf_cfg.pad_token_id = eos, eos, pad
    model = LlamaForCausalLM(hf_cfg)
    missing, unexpected = model.load_state_dict(to_hf_state_dict(state, cfg.n_layers), strict=False)
    assert not unexpected and set(missing) <= {"lm_head.weight"}, (missing, unexpected)
    model.tie_weights()
    model.eval().save_pretrained(out, safe_serialization=True)

    stop = [tok.special_to_id["</sql>"], eos] if sql else [eos]
    GenerationConfig(bos_token_id=eos, eos_token_id=stop, pad_token_id=pad, do_sample=False,
                     max_new_tokens=160 if sql else 256).save_pretrained(out)

    hf_tok = to_hf_tokenizer(tok)
    fast = PreTrainedTokenizerFast(tokenizer_object=hf_tok, bos_token="<|endoftext|>", eos_token="<|endoftext|>",
                                   pad_token="<|pad|>", unk_token=None, model_max_length=cfg.max_seq_len,
                                   clean_up_tokenization_spaces=False)
    fast.save_pretrained(out)
    print(f"exported {ckpt} -> {out} ({sum(p.numel() for p in model.parameters()):,} parameters)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sql", action="store_true", help="fine-tuned text-to-SQL model: also stop generation at </sql>")
    a = ap.parse_args()
    export(a.ckpt, a.out, a.sql)


if __name__ == "__main__":
    main()
