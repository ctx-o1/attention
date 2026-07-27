import jax
import numpy as np
import pytest
import torch
from einops import rearrange
from flax import nnx
from torch import nn

from mha import MultiHeadAttention
from mha_einsum import MultiHeadAttentionEinsum
from mha_jax import MultiHeadAttentionJax


@pytest.mark.parametrize(
    "implementation", [MultiHeadAttention, MultiHeadAttentionEinsum]
)
@pytest.mark.parametrize("is_causal", [False, True])
def test_torch_mha_matches_reference(implementation, is_causal):
    torch.manual_seed(0)
    d_model, num_heads = 12, 3

    custom = implementation(num_heads, d_model).double()
    reference_mha = nn.MultiheadAttention(
        d_model, num_heads, dropout=0.0, bias=False, batch_first=True
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


@pytest.mark.parametrize("is_causal", [False, True])
def test_jax_mha_matches_reference(is_causal):
    d_model, num_heads = 12, 3
    input_key = jax.random.key(1)

    custom = MultiHeadAttentionJax(
        num_heads=num_heads,
        d_model=d_model,
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(input_key, (2, 5, d_model))
    actual_output, _ = custom(x, is_causal=is_causal)

    Q = x @ custom.W_q.kernel[...]
    K = x @ custom.W_k.kernel[...]
    V = x @ custom.W_v.kernel[...]

    Q = rearrange(Q, "b l (h d) -> b l h d", h=num_heads)
    K = rearrange(K, "b l (h d) -> b l h d", h=num_heads)
    V = rearrange(V, "b l (h d) -> b l h d", h=num_heads)

    expected_output = jax.nn.dot_product_attention(
        Q, K, V, is_causal=is_causal
    )
    expected_output = rearrange(expected_output, "b l h d -> b l (h d)")
    expected_output = expected_output @ custom.W_o.kernel[...]

    np.testing.assert_allclose(actual_output, expected_output, rtol=1e-5, atol=1e-6)
