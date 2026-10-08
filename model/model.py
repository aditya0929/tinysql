import torch
import torch.nn as nn

from model.block import DecoderBlock
from model.config import ModelConfig
from model.embedding import Embedding
from model.linear import Linear
from model.rmsnorm import RMSNorm
from model.rope import RotaryEmbedding


class _CrossEntropy(torch.autograd.Function):
    """Memory-lean cross-entropy over a large vocabulary.

    The logits tensor is (tokens x 32,768), by far the biggest tensor in the model (2 GB in fp32 for 8 x 2048
    tokens). A naive version keeps several fp32 copies alive for the backward pass. This one saves only the
    original logits plus one number per token (the log-sum-exp), works in place, and recomputes the softmax in
    the backward pass:  d loss / d logits = (softmax(logits) - onehot(target)) / n_tokens.
    """

    @staticmethod
    def forward(ctx, logits, targets, ignore_index):
        flat = logits.reshape(-1, logits.shape[-1])
        t = targets.reshape(-1)
        keep = t != ignore_index
        safe = t.masked_fill(~keep, 0)
        target_logit = flat.gather(1, safe[:, None]).squeeze(1).float()
        x = flat.to(torch.float32, copy=True)                             # always a copy: the in-place ops below must not touch the input
        m = x.max(dim=-1, keepdim=True).values
        x.sub_(m).exp_()                                                  # in place: no extra full-size copies
        lse = m.squeeze(-1) + x.sum(dim=-1).log()                         # stable log(sum(exp(logits)))
        n = keep.sum().clamp(min=1)
        loss = ((lse - target_logit) * keep).sum() / n
        ctx.save_for_backward(flat, lse, safe, keep, n)
        ctx.shape = logits.shape
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        flat, lse, safe, keep, n = ctx.saved_tensors
        p = flat.to(torch.float32, copy=True)
        p.sub_(lse[:, None]).exp_()                                       # softmax, recomputed
        p.scatter_add_(1, safe[:, None], torch.full_like(p[:, :1], -1.0))  # minus one-hot at the target
        p.mul_((keep / n).float()[:, None] * grad_out)
        return p.to(flat.dtype).reshape(ctx.shape), None, None


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Mean negative log-likelihood of the target tokens. Positions equal to ignore_index are skipped
    (used to mask prompt tokens during fine-tuning)."""
    return _CrossEntropy.apply(logits, targets, ignore_index)


class TinySQL(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = Embedding(cfg.vocab_size, cfg.d_model, std=cfg.init_std)
        self.blocks = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = None if cfg.tie_embeddings else Linear(cfg.d_model, cfg.vocab_size, std=cfg.init_std)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)   # shared by all layers

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, ids, targets=None, start_pos: int = 0, past_kvs=None):
        """ids: (B, T) token ids. Returns (logits (B, T, vocab), loss or None, list of per-layer (k, v))."""
        assert start_pos + ids.shape[1] <= self.cfg.max_seq_len, "sequence longer than max_seq_len"
        x = self.embed(ids)
        new_kvs = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, self.rope, start_pos, None if past_kvs is None else past_kvs[i])
            new_kvs.append(kv)
        x = self.final_norm(x)
        logits = x @ self.embed.weight.T if self.lm_head is None else self.lm_head(x)   # tied head reuses E
        loss = None if targets is None else cross_entropy(logits, targets)
        return logits, loss, new_kvs
