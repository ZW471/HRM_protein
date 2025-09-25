from torch import nn
import torch

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        scores = (torch.bmm(q, k.transpose(-2, -1)))
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        # print(self.softmax(
        #     scores / torch.sqrt(torch.tensor(k.shape[-1]))
        # ))
        return torch.bmm(self.softmax(
            scores / torch.sqrt(torch.tensor(k.shape[-1]))
        ), v)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim_in, dim_qk, dim_v, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W_q = nn.Linear(dim_in, dim_qk, bias=False)
        self.W_k = nn.Linear(dim_in, dim_qk, bias=False)
        self.W_v = nn.Linear(dim_in, dim_v, bias=False)
        self.attention = Attention()

    def forward(self, x, mask=None):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        return self.attention(q, k, v, mask=mask)

def causal_mask(T: int, device=None, dtype=torch.bool):
    # True above the diagonal ⇒ blocked
    return torch.triu(torch.ones(T, T, dtype=dtype, device=device), diagonal=1).bool()

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_qk, dim_v, dim_out, num_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W_q = nn.ModuleList([nn.Linear(dim_in, dim_qk, bias=False) for _ in range(num_heads)])
        self.W_k = nn.ModuleList([nn.Linear(dim_in, dim_qk, bias=False) for _ in range(num_heads)])
        self.W_v = nn.ModuleList([nn.Linear(dim_in, dim_v, bias=False) for _ in range(num_heads)])

        self.W_out = nn.Linear(dim_v * num_heads, dim_out, bias=False)

        self.attention = Attention()

    def forward(self, x, mask=None):
        V = []
        for W_q, W_k, W_v in zip(self.W_q, self.W_k, self.W_v):
            v_head = self.attention(W_q(x), W_k(x), W_v(x), mask=mask)
            V.append(v_head)

        return self.W_out(torch.concat(V, dim=-1))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_in, dim_qk, dim_v, num_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mha = MultiHeadAttention(
            dim_in=dim_in,
            dim_qk=dim_qk,
            dim_v=dim_v,
            dim_out=dim_in,
            num_heads=num_heads,
        )

        self.mha_norm = nn.LayerNorm(dim_in)
        self.activation = nn.GELU()
        self.ffn = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            self.activation,
            nn.Linear(dim_in, dim_in)
        )
        self.ff_norm = nn.LayerNorm(dim_in)

    def forward(self, x):
        mha_out = self.mha_norm(self.mha(x)) + x
        return self.ff_norm(self.ffn(mha_out)) + mha_out

class TransformerEncoder(nn.Module):
    def __init__(self, dim_in, dim_qk, dim_v, num_heads, num_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers = [TransformerEncoderLayer(dim_in=dim_in, dim_qk=dim_qk, dim_v=dim_v, num_heads=num_heads) for _ in range(num_layers)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim_in, dim_qk, dim_v, num_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.masked_mha = MultiHeadAttention(
            dim_in=dim_in,
            dim_qk=dim_qk,
            dim_v=dim_v,
            dim_out=dim_in,
            num_heads=num_heads,
        )
        self.masked_mha_norm = nn.LayerNorm(dim_in)

        self.mha = MultiHeadAttention(
            dim_in=dim_in,
            dim_qk=dim_qk,
            dim_v=dim_v,
            dim_out=dim_in,
            num_heads=num_heads,
        )
        self.mha_norm = nn.LayerNorm(dim_in)

        self.activation = nn.GELU()
        self.ffn = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            self.activation,
            nn.Linear(dim_in, dim_in)
        )
        self.ff_norm = nn.LayerNorm(dim_in)

    def forward(self, x, mask):
        masked_mha_out = self.masked_mha_norm(self.masked_mha(x, mask=mask)) + x
        mha_out = self.mha_norm(self.mha(masked_mha_out)) + masked_mha_out
        return self.ff_norm(self.ffn(mha_out)) + mha_out

class TransformerDecoder(nn.Module):
    def __init__(self, dim_in, dim_qk, dim_v, num_heads, num_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers = [TransformerDecoderLayer(dim_in=dim_in, dim_qk=dim_qk, dim_v=dim_v, num_heads=num_heads) for _ in range(num_layers)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x