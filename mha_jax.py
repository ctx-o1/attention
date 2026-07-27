import jax
import jax.numpy as jnp
from einops import einsum, rearrange
from flax import nnx


def generate_causal_mask(sequence_length: int) -> jax.Array:
    """Return a boolean mask where True marks future positions."""
    return jnp.triu(jnp.ones((sequence_length, sequence_length), dtype=bool), k=1)


class MultiHeadAttentionJax(nnx.Module):
    def __init__(self, num_heads: int, d_model: int, rngs: nnx.Rngs):
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.W_q = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)
        self.W_k = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)
        self.W_v = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)
        self.W_o = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array, is_causal: bool = False):
        _, sequence_length, _ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = rearrange(
            Q, "b l (h d) -> b h l d", h=self.num_heads
        )
        K = rearrange(
            K, "b l (h d) -> b h l d", h=self.num_heads
        )
        V = rearrange(
            V, "b l (h d) -> b h l d", h=self.num_heads
        )

        scores = einsum(
            Q, K, "b h q d, b h k d -> b h q k"
        ) / jnp.sqrt(self.head_dim)

        if is_causal:
            causal_mask = generate_causal_mask(sequence_length)
            scores = jnp.where(causal_mask, -jnp.inf, scores)

        attention = jax.nn.softmax(scores, axis=-1)
        out = einsum(
            attention, V, "b h q k, b h k d -> b h q d"
        )
        out = rearrange(out, "b h l d -> b l (h d)")
        out = self.W_o(out)

        return out, attention
