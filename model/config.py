from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 32_768
    d_model: int = 576          # size of each token's vector
    n_layers: int = 30
    n_heads: int = 9            # query heads
    n_kv_heads: int = 3         # key/value heads (grouped-query attention)
    d_ff: int = 1536            # SwiGLU hidden size
    max_seq_len: int = 2048
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    init_std: float = 0.02      # std of the normal distribution used to initialise weights
    fused_attention: bool = True    # PyTorch fused kernel (fast); False = my manual attention

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must split evenly into heads"
        assert self.n_heads % self.n_kv_heads == 0, "query heads must split evenly into KV groups"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_rep(self) -> int:
        """How many query heads share each key/value head."""
        return self.n_heads // self.n_kv_heads

    def expected_params(self) -> int:
        """Parameter count derived by hand (matches the table in TINYSQL_PLAN.md)."""
        d, kv = self.d_model, self.n_kv_heads * self.head_dim
        attn = d * d + d * kv + d * kv + d * d      # Q, K, V, O
        mlp = 3 * d * self.d_ff                     # gate, up, down
        norms = 2 * d                               # two RMSNorms per layer
        per_layer = attn + mlp + norms
        total = self.vocab_size * d + self.n_layers * per_layer + d   # + final norm
        if not self.tie_embeddings:
            total += self.vocab_size * d
        return total
