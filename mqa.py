import math
import torch
from torch import nn
import torch.nn.functional as F

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

        Q = Q.view(B, L, self.h, self.d_k).transpose(1, 2)  # (B, h, L, d_k)
        K = K.unsqueeze(1)  # (B, 1, L, d_k)
        V = V.unsqueeze(1)  # (B, 1, L, d_k)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (B, h, L, L)

        if is_causal:
            mask = generate_causal_mask(L, device=x.device)
            scores = scores.masked_fill(mask, float("-inf"))

        attention = F.softmax(scores, dim=-1)
        out = attention @ V  # (B, h, L, d_k)

        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.W_o(out)  # (B, L, d_model)

        return out, attention
