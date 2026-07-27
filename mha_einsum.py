import math

import torch
from einops import einsum, rearrange
from torch import nn

from utils import generate_causal_mask


class MultiHeadAttentionEinsum(nn.Module):
    def __init__(self, h, d_model):
        super().__init__()

        assert d_model % h == 0

        self.h = h
        self.d_model = d_model
        self.d_k = d_model // h

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, is_causal: bool = False):
        _, sequence_length, _ = x.shape

        Q = rearrange(
            self.W_q(x), "b l (h d) -> b h l d", h=self.h
        )
        K = rearrange(
            self.W_k(x), "b l (h d) -> b h l d", h=self.h
        )
        V = rearrange(
            self.W_v(x), "b l (h d) -> b h l d", h=self.h
        )

        scores = einsum(
            Q, K, "b h q d, b h k d -> b h q k"
        ) / math.sqrt(self.d_k)

        if is_causal:
            causal_mask = generate_causal_mask(sequence_length, device=x.device)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        attention = torch.softmax(scores, dim=-1)
        out = einsum(
            attention, V, "b h q k, b h k d -> b h q d"
        )
        out = rearrange(out, "b h l d -> b l (h d)")

        return self.W_o(out), attention
