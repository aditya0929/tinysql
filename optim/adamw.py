import torch


class AdamW:
    """AdamW with decoupled weight decay, written from scratch (checked against torch.optim.AdamW in tests).

    Per parameter p with gradient g, at step t:
        p <- p * (1 - lr * weight_decay)          decay acts on the weights directly, not through the gradient
        m <- b1 * m + (1 - b1) * g                running mean of gradients
        v <- b2 * v + (1 - b2) * g^2              running mean of squared gradients
        p <- p - lr * (m / (1 - b1^t)) / (sqrt(v / (1 - b2^t)) + eps)
    m and v are kept in fp32.
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.95), eps: float = 1e-8, weight_decay: float = 0.1):
        params = list(params)
        groups = params if params and isinstance(params[0], dict) else [{"params": params}]
        self.param_groups = []
        for g in groups:
            g = dict(g)
            g["params"] = list(g["params"])
            g.setdefault("lr", lr)
            g.setdefault("weight_decay", weight_decay)
            self.param_groups.append(g)
        self.betas, self.eps = betas, eps
        self.state = {}                     # keyed by (group index, param index) so checkpoints need no object ids
        self.step_count = 0

    def zero_grad(self):
        for g in self.param_groups:
            for p in g["params"]:
                p.grad = None

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        t, (b1, b2) = self.step_count, self.betas
        bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
        for gi, group in enumerate(self.param_groups):
            lr, wd = group["lr"], group["weight_decay"]
            for pi, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                g = p.grad.float()
                st = self.state.get((gi, pi))
                if st is None:
                    st = self.state[(gi, pi)] = {"m": torch.zeros_like(p, dtype=torch.float32),
                                                 "v": torch.zeros_like(p, dtype=torch.float32)}
                m, v = st["m"], st["v"]
                if wd:
                    p.mul_(1 - lr * wd)
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                denom = (v / bc2).sqrt_().add_(self.eps)
                p.addcdiv_((m / bc1).to(p.dtype), denom.to(p.dtype), value=-lr)

    def state_dict(self):
        return {"step": self.step_count, "betas": self.betas, "eps": self.eps,
                "lrs": [g["lr"] for g in self.param_groups],
                "state": {f"{gi}.{pi}": {k: v.clone() for k, v in st.items()} for (gi, pi), st in self.state.items()}}

    def load_state_dict(self, sd):
        self.step_count, self.betas, self.eps = sd["step"], tuple(sd["betas"]), sd["eps"]
        for g, lr in zip(self.param_groups, sd["lrs"]):
            g["lr"] = lr
        self.state = {}
        for key, st in sd["state"].items():
            gi, pi = map(int, key.split("."))
            ref = self.param_groups[gi]["params"][pi]
            self.state[(gi, pi)] = {k: v.to(ref.device) for k, v in st.items()}


@torch.no_grad()
def clip_grad_norm_(params, max_norm: float) -> torch.Tensor:
    """Scale all gradients so their global L2 norm is at most max_norm. Returns the norm before clipping
    (may be inf or nan: the caller should then skip the optimizer step)."""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    total = torch.sqrt(sum(g.float().pow(2).sum() for g in grads))
    if torch.isfinite(total):
        scale = max_norm / (total + 1e-6)
        if scale < 1:
            for g in grads:
                g.mul_(scale.to(g.dtype))
    return total
