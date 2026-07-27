# Implemetation Notes

## Flax RNG Notes

JAX uses explicit random keys instead of a global random state. Flax NNX manages these keys with `nnx.Rngs`:

```python
rngs = nnx.Rngs(0)
```

Layers use the RNG stream to initialize their parameters:

```python
self.W_q = nnx.Linear(
    in_features=d_model,
    out_features=d_model,
    use_bias=False,
    rngs=rngs,
)
```

Passing the same `rngs` object to multiple layers is safe. NNX produces a fresh subkey for each initialization, so the layers receive different weights.

Named streams can separate different uses of randomness:

```python
rngs = nnx.Rngs(params=0, dropout=1)
```

Our MHA only needs an RNG when its weights are initialized because its forward pass has no dropout.

## Matmul broadcasting and `unsqueeze`

`unsqueeze(dim)` inserts a new dimension of size `1` without copying tensor data:

```python
x = torch.randn(2, 5, 4)  # (2, 5, 4)
y = x.unsqueeze(1)        # (2, 1, 5, 4)
```

`squeeze(dim)` removes a size-one dimension.

`torch.matmul` uses the last two dimensions as matrices and broadcasts all preceding dimensions from right to left.

In MQA:

```text
Q: (B, h, L, d_k)
K: (B,    L, d_k)
```

The batch dimensions `(B, h)` and `(B)` do not align correctly. Add a size-one head dimension:

```python
K = K.unsqueeze(1)  # (B, 1, L, d_k)
V = V.unsqueeze(1)  # (B, 1, L, d_k)
```

Now `(B, 1)` broadcasts to `(B, h)` without copying K or V:

```python
scores = Q @ K.transpose(-2, -1)  # (B, h, L, L)
out = attention @ V               # (B, h, L, d_k)
```

Broadcasting dry run for `B=2`, `h=3`, `L=5`, and `d_k=4`:

```text
Q:              (2, 3, 5, 4)
K:              (2, 1, 5, 4)
K transpose:    (2, 1, 4, 5)

matrix multiply: (5, 4) @ (4, 5) -> (5, 5)
batch broadcast: (2, 3) and (2, 1) -> (2, 3)

scores:         (2, 3, 5, 5)
```

Conceptually, each query head uses the same K head:

```python
scores[b, head] = Q[b, head] @ K[b, 0].T
```

Expanded for two batches and three query heads:

```text
batch 0:
  Q[0, 0] @ K[0, 0].T -> scores[0, 0]
  Q[0, 1] @ K[0, 0].T -> scores[0, 1]
  Q[0, 2] @ K[0, 0].T -> scores[0, 2]

batch 1:
  Q[1, 0] @ K[1, 0].T -> scores[1, 0]
  Q[1, 1] @ K[1, 0].T -> scores[1, 1]
  Q[1, 2] @ K[1, 0].T -> scores[1, 2]
```

The batch dimension still selects a different K for each input. Only the head dimension is shared. The singleton head dimension does not copy K; it tells broadcasting to reuse `K[b, 0]` for every query head in batch `b`.

## Einsum

General `einops.einsum` syntax:

```python
einsum(
    tensor_1,
    tensor_2,
    "tensor_1_axes, tensor_2_axes -> output_axes",
)
```

Einsum does not infer a matrix multiplication from the tensor shapes. It follows the named axes. The main rule is:

> Any input axis omitted from the output is multiplied over and summed away.

For the MQA value calculation:

```python
out = einsum(
    attention,
    V,
    "b h query key, b key d -> b h query d",
)
```

```text
attention: b h query key
V:         b         key d
output:    b h query     d
```

- `b` is shared by both inputs and retained.
- `key` is shared but omitted from the output, so it is summed over.
- `h` and `query` come from `attention`.
- `d` comes from `V`.

Mathematically:

```text
out[b, h, query, d]
    = sum over key of attention[b, h, query, key] * V[b, key, d]
```

Because V has no head axis, the same V is used by every query head. No `unsqueeze` is required.

Useful patterns:

```python
# Dot product: sum over d
einsum(a, b, "d, d ->")

# Matrix multiplication: sum over k
einsum(A, B, "m k, k n -> m n")

# Batched matrix multiplication
einsum(A, B, "b m k, b k n -> b m n")

# Outer product: no axis is removed, so there is no summation
einsum(a, b, "i, j -> i j")

# Sum over sequence axis l
einsum(x, "b l d -> b d")
```

Mental model:

1. Label every input dimension.
2. Align and multiply dimensions with matching labels.
3. Sum labels omitted from the output.
4. Arrange the remaining dimensions in output order.


Use `rearrange` for reshaping and transposing, and `einsum` when dimensions are multiplied and reduced.


---

- [Flax Linen to NNX comparison](https://flax.readthedocs.io/en/latest/migrating/linen_to_nnx.html).
