import sys, time
sys.path.insert(0, ".")
import torch
from model.config import ModelConfig
from model.model import TinySQL
from model.generate import generate, filter_logits
from model.attention import softmax

torch.manual_seed(0)
model = TinySQL(ModelConfig()).eval()
prompt = torch.randint(0, 32768, (1, 16))
N = 48

t = time.time(); fast = generate(model, prompt, N, temperature=0); t_cache = time.time() - t

t = time.time(); ids = prompt
with torch.no_grad():
    for _ in range(N):                       # no cache: recompute everything each step
        logits, _, _ = model(ids)
        ids = torch.cat([ids, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
t_nocache = time.time() - t
print(f"{N} tokens | with KV cache: {t_cache:.2f}s | without: {t_nocache:.2f}s | same output: {torch.equal(fast, ids)}")

# What temperature / top-k / top-p do to one distribution
logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0, -1.0]])
for name, l in [("raw", logits), ("T=0.5", logits / 0.5), ("T=2.0", logits / 2.0),
                ("top_k=2", filter_logits(logits, top_k=2)), ("top_p=0.9", filter_logits(logits, top_p=0.9))]:
    print(f"{name:10s}", [round(p, 3) for p in softmax(l.float(), dim=-1)[0].tolist()])
