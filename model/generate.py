import torch

from model.attention import softmax


def filter_logits(logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None) -> torch.Tensor:
    """Set the logits of unwanted tokens to -inf. logits: (B, V).

    top_k: keep only the k highest-scoring tokens.
    top_p: keep the smallest set of top tokens whose probabilities add up to at least p ("nucleus").
    """
    if top_k is None and top_p is None:
        return logits
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
    remove = torch.zeros_like(sorted_logits, dtype=torch.bool)
    if top_k is not None:
        remove[:, top_k:] = True
    if top_p is not None:
        probs = softmax(sorted_logits.float(), dim=-1)
        mass_before = probs.cumsum(dim=-1) - probs          # probability mass of all higher-ranked tokens
        remove |= mass_before > top_p                       # the top token has mass_before = 0, so it always stays
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)   # undo the sort


def sample_from_probs(probs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Draw one index per row from a probability distribution (inverse-CDF sampling).
    probs: (B, V), rows sum to 1. Returns (B,)."""
    cdf = probs.cumsum(dim=-1)
    u = torch.rand(probs.shape[0], 1, generator=generator, device=probs.device)    # uniform in [0, 1)
    target = u * cdf[:, -1:]                                                       # robust to rounding in the sum
    return (cdf <= target).sum(dim=-1).clamp(max=probs.shape[-1] - 1)              # first index where cdf > target


def sample_next(logits, temperature: float = 1.0, top_k=None, top_p=None, generator=None) -> torch.Tensor:
    """logits: (B, V) -> next token ids (B,). temperature == 0 means greedy (argmax)."""
    logits = logits.float()
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = filter_logits(logits / temperature, top_k, top_p)
    return sample_from_probs(softmax(logits, dim=-1), generator)


@torch.no_grad()
def generate(model, prompt_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
             top_k=None, top_p=None, stop_ids=(), generator=None) -> torch.Tensor:
    """Autoregressive generation with a KV cache. prompt_ids: (B, T). Returns (B, T + n_generated).

    A sequence that emits any id in stop_ids is finished; it is padded with that stop id until
    every sequence in the batch has finished (or max_new_tokens is reached)."""
    B, T = prompt_ids.shape
    max_new_tokens = min(max_new_tokens, model.cfg.max_seq_len - T)
    stop = torch.tensor(list(stop_ids), dtype=torch.long, device=prompt_ids.device)
    finished = torch.zeros(B, dtype=torch.bool, device=prompt_ids.device)
    ids = prompt_ids

    logits, _, kvs = model(prompt_ids)                        # prefill: process the whole prompt once
    for step in range(max_new_tokens):
        tok = sample_next(logits[:, -1], temperature, top_k, top_p, generator)
        if len(stop):
            tok = torch.where(finished, stop[0], tok)         # finished sequences just repeat the stop id
            finished |= torch.isin(tok, stop)
        ids = torch.cat([ids, tok[:, None]], dim=1)
        if finished.all() or step == max_new_tokens - 1:
            break
        logits, _, kvs = model(tok[:, None], start_pos=ids.shape[1] - 1, past_kvs=kvs)   # one token at a time
    return ids
