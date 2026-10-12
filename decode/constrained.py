"""Grammar- and schema-constrained greedy decoding.

At every step the model's tokens are tried best-first; the first one that keeps the output a valid prefix of the
SQL grammar (decode/sql_grammar.py) is taken. Because the grammar only admits real column names (and, optionally,
values copied from the question), the model cannot hallucinate a column or emit broken SQL. Once the query is
complete the only allowed continuation is </sql>.

Greedy over the allowed set equals "mask every disallowed token to -inf, then argmax", but checking candidates
lazily in score order avoids testing all 32,768 tokens at every step.
"""
import itertools
from collections import defaultdict

import torch

from data.wikisql_to_sql import column_names
from decode.sql_grammar import SqlGrammar


def _choose(logits_row, text: bytes, grammar: SqlGrammar, tok, stop_id: int, topk: int):
    if grammar.check(text) == "complete":
        return stop_id
    top = torch.topk(logits_row, topk).indices.tolist()
    for t in itertools.chain(top, torch.argsort(logits_row, descending=True).tolist()):
        if t in tok.id_to_special:
            continue
        if grammar.check(text + tok.vocab[t]) != "invalid":
            return t
    return None                                               # nothing fits: give up on this row


@torch.no_grad()
def constrained_generate(model, tok, prompts, grammars, max_new_tokens: int = 160, device="cpu", topk: int = 64):
    """prompts: equal-length token lists (one group). Returns the generated tokens per prompt, without the stop token."""
    stop_id = tok.special_to_id["</sql>"]
    ids = torch.tensor(prompts, dtype=torch.long, device=device)
    T = ids.shape[1]
    logits, _, kvs = model(ids)
    last = logits[:, -1].float().cpu()
    out = [[] for _ in prompts]
    text = [b"" for _ in prompts]
    done = [False] * len(prompts)
    for step in range(min(max_new_tokens, model.cfg.max_seq_len - T)):
        nxt = []
        for b in range(len(prompts)):
            if done[b]:
                nxt.append(stop_id)
                continue
            t = _choose(last[b], text[b], grammars[b], tok, stop_id, topk)
            if t is None or t == stop_id:
                done[b] = True
                nxt.append(stop_id)
                continue
            out[b].append(t)
            text[b] += tok.vocab[t]
            nxt.append(t)
        if all(done):
            break
        logits, _, kvs = model(torch.tensor(nxt, device=device)[:, None], start_pos=T + step, past_kvs=kvs)
        last = logits[:, -1].float().cpu()
    return out


def make_constrained_generator(model, tok, items, tables, values_from_question: bool = True, device="cpu",
                               max_batch: int = 64, max_new_tokens: int = 160):
    """Returns generator(prompts, stop_id) for eval.evaluate_sql.evaluate; items/tables give each prompt its grammar."""
    def generator(prompts, _stop):
        grammars = []
        for it in items:
            t = tables[it["table_id"]]
            grammars.append(SqlGrammar(column_names(t["header"]), it["question"] if values_from_question else None))
        out = [None] * len(prompts)
        by_len = defaultdict(list)
        for i, p in enumerate(prompts):
            by_len[len(p)].append(i)
        model.eval()
        for idxs in by_len.values():
            for s in range(0, len(idxs), max_batch):
                chunk = idxs[s:s + max_batch]
                gen = constrained_generate(model, tok, [prompts[i] for i in chunk], [grammars[i] for i in chunk],
                                           max_new_tokens, device)
                for i, g in zip(chunk, gen):
                    out[i] = g
        return out
    return generator
