import sys, time, glob, sysconfig, os
sys.path.insert(0, ".")
from tokenizer.bpe import BPETokenizer

lib = sysconfig.get_paths()["stdlib"]
files = sorted(glob.glob(os.path.join(lib, "*.py")))
texts, size = [], 0
for f in files:
    try: t = open(f, encoding="utf-8").read()
    except Exception: continue
    texts.append(t); size += len(t)
texts.append(open("TINYSQL_PLAN.md", encoding="utf-8").read())
held_out, train_texts = texts[::10], [t for i, t in enumerate(texts) if i % 10]
print(f"{len(texts)} files, {size/1e6:.1f} MB total; training on {len(train_texts)}")

V = 4000
t0 = time.time()
tok = BPETokenizer.train(train_texts, vocab_size=V)
print(f"trained vocab {tok.vocab_size} in {time.time()-t0:.1f}s")

print("first 12 merges :", [tok.vocab[256+i].decode("utf-8", "replace") for i in range(12)])
print("merges 1000-1008:", [tok.vocab[256+i].decode("utf-8", "replace") for i in range(1000, 1008)])

s = "SELECT customer_id, SUM(total) FROM orders GROUP BY customer_id;"
ids = tok.encode(s)
print(f"\n{len(s)} chars -> {len(ids)} tokens:", [tok.decode([i]) for i in ids])

held = "".join(held_out); n_bytes = len(held.encode())
mine = len(tok.encode(held))
print(f"\nheld-out: {n_bytes/mine:.2f} bytes/token (mine)")

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
hf = Tokenizer(models.BPE())
hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
hf.decoder = decoders.ByteLevel()
hf.train_from_iterator(train_texts, trainers.BpeTrainer(vocab_size=V - len(tok.special_tokens),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
hf_n = len(hf.encode(held).ids)
print(f"held-out: {n_bytes/hf_n:.2f} bytes/token (Hugging Face tokenizers, same data & vocab size)")
