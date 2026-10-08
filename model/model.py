import torch
import torch.nn as nn

from model.block import DecoderBlock
from model.config import ModelConfig
from model.embedding import Embedding
from model.linear import Linear
from model.rmsnorm import RMSNorm
from model.rope import RotaryEmbedding


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Mean negative log-likelihood of the target tokens. Positions equal to ignore_index are skipped
    (used to mask prompt tokens during fine-tuning)."""
    logits = logits.float().reshape(-1, logits.shape[-1])                  # (N, V), fp32
    targets = targets.reshape(-1)
    m = logits.max(dim=-1, keepdim=True).values
    logsumexp = m.squeeze(-1) + (logits - m).exp().sum(dim=-1).log()       # stable log(sum(exp))
    keep = targets != ignore_index
    safe_targets = targets.masked_fill(~keep, 0)
    target_logit = logits.gather(1, safe_targets[:, None]).squeeze(1)
    nll = (logsumexp - target_logit) * keep                                # -log softmax(logits)[target]
    return nll.sum() / keep.sum().clamp(min=1)


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
