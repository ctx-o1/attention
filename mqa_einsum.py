import math
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, einsum

from utils import generate_causal_mask


class MultiQueryAttention(nn.Module):
    def __init__(self, h, d_model):
        super().__init__()

        assert d_model % h == 0

        self.h = h
        self.d_model = d_model
        self.d_k = d_model // h

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, self.d_k, bias=False)

        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, is_causal: bool = True):
        B, L, _ = x.size()

        Q = self.W_q(x)  # (B, L, d_model)
        K = self.W_k(x)  # (B, L, d_k)
        V = self.W_v(x)  # (B, L, d_k)

        Q = rearrange(Q, "b l (h d) -> b h l d", h=self.h)
        K = rearrange(K, "b l d -> b 1 l d")
        V = rearrange(V, "b l d -> b 1 l d")

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (B, h, L, L)

        if is_causal:
            mask = generate_causal_mask(L, device=x.device)
            scores = scores.masked_fill(mask, float("-inf"))

        attention = F.softmax(scores, dim=-1)
        out = attention @ V  # (B, h, L, d_k)

        out = rearrange(out, "b h l d -> b l (h d)")
        out = self.W_o(out)  # (B, L, d_model)

        return out, attention
