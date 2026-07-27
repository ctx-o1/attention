import math

import torch
from torch import nn

from utils import generate_causal_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model):
        super().__init__()

        assert d_model % h == 0  # d_k = d_v = d_model / num_heads

        self.h = h
        self.d_model = d_model
        self.d_k = d_model // h

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, is_causal: bool = False):

        # x: (B, L, d_model)
        B, L, _ = x.size()

        # project first
        Q = self.W_q(x)  # (B, L, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # split heads
        Q = Q.view(B, L, self.h, self.d_k)  # (B, L, h, d_k)
        Q = Q.transpose(1, 2)  # (B, h, L, d_k)

        K = K.view(B, L, self.h, self.d_k).transpose(1, 2)
        V = V.view(B, L, self.h, self.d_k).transpose(1, 2)

        # Q @ K_T / root d_k
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (B, h, L, L)

        if is_causal:
            causal_mask = generate_causal_mask(L, device=x.device)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        attention = torch.softmax(scores, dim=-1)  # (B, h, L, L)
        out = attention @ V  # (B, h, L, d_k)

        # concat (B, L, d_model)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)

        out = self.W_o(out)  # (B, L, d_model)

        return out, attention
