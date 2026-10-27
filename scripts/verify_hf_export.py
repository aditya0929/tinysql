"""End-to-end check that an exported Hugging Face folder behaves exactly like the from-scratch model.

    python -m scripts.verify_hf_export --ckpt checkpoints/tinysql_125m_wikisql_lr5e-05_s0.pt --hf release/TinySQL-125M-WikiSQL
"""
import argparse
import os
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.wikisql_to_sql import BASE, load_split, prompt_text, to_sql
from eval.evaluate_sql import generate_grouped, score_predictions
from eval.execution_acc import WikiSQLExecutor
from model.config import ModelConfig
from model.model import TinySQL
from tokenizer.bpe import BPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    mine_tok = BPETokenizer.load("tokenizer/tinysql_bpe.json")
    hf_tok = AutoTokenizer.from_pretrained(a.hf)

    texts = []
    for split in ("dev", "test"):
        items, tables = load_split(split)
        for it in items:
            t = tables[it["table_id"]]
            texts += [prompt_text(it["question"], t["header"], t["types"]), to_sql(it["sql"], t["header"])]
    bad = sum(hf_tok(x, add_special_tokens=False).input_ids != mine_tok.encode(x) for x in texts)
    print(f"tokenizer: {len(texts):,} strings, {bad} differ")

    c = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    mine = TinySQL(ModelConfig(**c["cfg"]["model"])).eval()
    mine.load_state_dict(c["model"])
    hf = AutoModelForCausalLM.from_pretrained(a.hf, attn_implementation="eager").eval()

    items, tables = load_split("dev")
    items = items[:a.n]
    prompts = [mine_tok.encode(prompt_text(it["question"], tables[it["table_id"]]["header"], tables[it["table_id"]]["types"]))
               for it in items]
    stop = mine_tok.special_to_id["</sql>"]
    ours = generate_grouped(mine, prompts, [stop, mine_tok.special_to_id["<|endoftext|>"]])
    theirs = [None] * len(prompts)
    by_len = defaultdict(list)
    for i, p in enumerate(prompts):
        by_len[len(p)].append(i)
    with torch.no_grad():
        for idxs in by_len.values():                                       # equal lengths: no padding needed
            out = hf.generate(torch.tensor([prompts[i] for i in idxs]), max_new_tokens=160, do_sample=False)
            for i, row in zip(idxs, out):
                g = row[len(prompts[i]):].tolist()
                for s in (stop, mine_tok.special_to_id["<|endoftext|>"]):
                    g = g[:g.index(s)] if s in g else g
                theirs[i] = g
    same = sum(o == t for o, t in zip(ours, theirs))
    ex = WikiSQLExecutor(os.path.join(BASE, "dev.db"), tables)
    m1, _ = score_predictions(items, tables, ex, [mine_tok.decode(g).strip() for g in ours])
    m2, _ = score_predictions(items, tables, ex, [hf_tok.decode(g).strip() for g in theirs])
    print(f"generation: {same}/{len(items)} identical; execution accuracy ours {m1['exec_acc']:.4f}, HF {m2['exec_acc']:.4f}")


if __name__ == "__main__":
    main()
