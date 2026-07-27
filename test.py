import pytest
import torch
from torch import nn

from mha import MultiHeadAttention


@pytest.mark.parametrize("is_causal", [False, True])
def test_mha_matches_reference(is_causal):
    torch.manual_seed(0)
    d_model, num_heads = 12, 3

    custom = MultiHeadAttention(num_heads, d_model).double()
    reference_mha = nn.MultiheadAttention(
        d_model, num_heads, dropout=0.0, batch_first=True
    ).double()

    # PyTorch stores the Q, K, and V projections in single packed tensors.
    with torch.no_grad():
        reference_mha.in_proj_weight.copy_(
            torch.cat([custom.W_q.weight, custom.W_k.weight, custom.W_v.weight])
        )
        reference_mha.out_proj.weight.copy_(custom.W_o.weight)

    x = torch.randn(2, 5, d_model, dtype=torch.float64)
    causal_mask = None
    if is_causal:
        causal_mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)

    actual_output, actual_attention = custom(x, is_causal=is_causal)
    expected_output, expected_attention = reference_mha(
        query=x,
        key=x,
        value=x,
        attn_mask=causal_mask,
        need_weights=True,  # return the attention matrix alongside its output
        average_attn_weights=False,  # keeps attention weights separate for each head
    )

    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(actual_attention, expected_attention)
