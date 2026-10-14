"""Run OUR constrained decoder (decode/constrained.py) on a Hugging Face baseline, so constraints are compared fairly.

    python -m eval.baselines.constrained_hf --model_dir runs/baseline_smollm2-135m/best --decode values --out runs/x

Two small adapters make a Hugging Face model look like TinySQL to the decoder:
  HFTokenBytes   token id -> raw bytes (byte-level BPE tokens are stored as printable stand-ins for bytes);
  HFModelAdapter model(ids) / model(x, start_pos, past_kvs) -> (logits, None, cache), using HF's own KV cache.
Prompts in a group have identical length, so no attention mask is needed (the same reason as for our model).
"""
import argparse
import json
import os
import time
from collections import defaultdict
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.wikisql_to_sql import BASE, column_names, load_split, prompt_text
from decode.constrained import constrained_generate
from decode.sql_grammar import SqlGrammar
from eval.evaluate_sql import score_predictions
from eval.execution_acc import WikiSQLExecutor


def bytes_to_unicode():
    """The standard byte-level BPE table: each of the 256 bytes is shown as a printable character in token strings."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class HFTokenBytes:
    """The parts of our tokenizer interface the constrained decoder uses: .vocab, .id_to_special, .special_to_id."""

    def __init__(self, tok):
        byte_of = {ch: b for b, ch in bytes_to_unicode().items()}
        special = set(tok.all_special_ids) | {i for i in tok.added_tokens_decoder}
        self.vocab, self.id_to_special = {}, {}
        for i in range(len(tok)):
            piece = tok.convert_ids_to_tokens(i)
            if i in special or piece is None or any(ch not in byte_of for ch in piece):
                self.id_to_special[i] = piece                  # never generated under the grammar
            else:
                self.vocab[i] = bytes(byte_of[ch] for ch in piece)
        stop = tok.eos_token_id
        self.special_to_id = {"</sql>": stop, "<|endoftext|>": stop}


class HFModelAdapter:
    def __init__(self, model):
        self.model = model
        self.cfg = SimpleNamespace(max_seq_len=model.config.max_position_embeddings)

    def eval(self):
        self.model.eval()
        return self

    @torch.no_grad()
    def __call__(self, ids, targets=None, start_pos=0, past_kvs=None):
        out = self.model(input_ids=ids, past_key_values=past_kvs, use_cache=True)
        return out.logits, None, out.past_key_values


@torch.no_grad()
def run(model_dir, decode, out_dir, split="dev", limit=None, max_batch=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = HFModelAdapter(AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).to(device)).eval()
    vocab = HFTokenBytes(tok)
    items, tables = load_split(split)
    items = items[:limit] if limit else items
    executor = WikiSQLExecutor(os.path.join(BASE, f"{split}.db"), tables)

    prompts = [tok(prompt_text(it["question"], tables[it["table_id"]]["header"], tables[it["table_id"]]["types"]),
                   add_special_tokens=False)["input_ids"] for it in items]
    grammars = [SqlGrammar(column_names(tables[it["table_id"]]["header"]), it["question"] if decode == "values" else None)
                for it in items]
    t0, gens = time.time(), [None] * len(items)
    by_len = defaultdict(list)
    for i, p in enumerate(prompts):
        by_len[len(p)].append(i)
    for idxs in by_len.values():
        for s in range(0, len(idxs), max_batch):
            chunk = idxs[s:s + max_batch]
            outs = constrained_generate(model, vocab, [prompts[i] for i in chunk], [grammars[i] for i in chunk], device=device)
            for i, o in zip(chunk, outs):
                gens[i] = o
    preds = [b"".join(vocab.vocab[t] for t in g).decode("utf-8", errors="replace") for g in gens]
    metrics, records = score_predictions(items, tables, executor, preds)
    metrics.update({"decode": decode, "model_dir": model_dir, "seconds": round(time.time() - t0, 1)})
    os.makedirs(out_dir, exist_ok=True)
    json.dump(metrics, open(os.path.join(out_dir, f"{split}_{decode}_metrics.json"), "w"), indent=1)
    with open(os.path.join(out_dir, f"{split}_{decode}_predictions.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{split} [{decode}] {model_dir}: {metrics['n']:,} examples | execution accuracy {metrics['exec_acc']:.2%} | "
          f"validity {metrics['validity']:.2%} | exact match {metrics['exact_match']:.2%} | {metrics['seconds']:.0f}s", flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--decode", default="values", choices=["schema", "values"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.model_dir, a.decode, a.out, split=a.split, limit=a.limit)


if __name__ == "__main__":
    main()
