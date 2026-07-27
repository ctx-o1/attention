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


---

- [Flax Linen to NNX comparison](https://flax.readthedocs.io/en/latest/migrating/linen_to_nnx.html).
