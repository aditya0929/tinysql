import torch

# 1. A tensor is a grid of numbers. The number of dimensions varies.
scalar = torch.tensor(3.0)                       # 0-D: one number
vector = torch.tensor([1.0, 2.0, 3.0])           # 1-D: a list
matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2-D: a table
print("scalar", scalar.shape, "| vector", vector.shape, "| matrix", matrix.shape)

# 2. In our model, a token is a vector of 576 numbers; a sequence is a stack of them;
#    a batch is a stack of sequences -> 3-D.
x = torch.randn(8, 2048, 576)    # (batch, seq_len, d_model)
print("model activations:", x.shape, "->", x.numel(), "numbers")

# 3. dtype = how each number is stored.
print("fp32 bytes/number:", torch.zeros(1).element_size())
print("bf16 bytes/number:", torch.zeros(1, dtype=torch.bfloat16).element_size())
print("bf16 cannot store 257:", torch.tensor(257.0).to(torch.bfloat16).item())

# 4. Elementwise ops act on every number independently.
print("elementwise:", vector * 2, vector ** 2)

# 5. Reductions collapse a dimension. This is exactly what RMSNorm does.
print("mean over rows:", matrix.mean(dim=0), "| over cols:", matrix.mean(dim=1))
tok = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
print("mean of squares:", tok.pow(2).mean(dim=-1, keepdim=True))  # 7.5

# 6. Broadcasting: shapes are stretched to match when one has size 1.
print("broadcast:", matrix / torch.tensor([[1.0], [2.0]]))   # row 1 / 1, row 2 / 2

# 7. Matrix multiply: the core operation of a neural network.
#    (2x3) @ (3x2) -> (2x2): each output = dot product of a row and a column.
A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
B = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
print("A @ B =\n", A @ B)

# 8. A 'linear layer' is exactly this: output = input @ weights.
tokens = torch.randn(4, 576)      # 4 tokens
W = torch.randn(576, 192)         # a learned weight matrix (like our K projection)
print("linear layer:", tokens.shape, "@", W.shape, "->", (tokens @ W).shape)

# 9. Indexing and reshaping: attention splits 576 numbers into 9 heads of 64.
h = torch.randn(2, 5, 576)
print("heads view:", h.view(2, 5, 9, 64).shape)
