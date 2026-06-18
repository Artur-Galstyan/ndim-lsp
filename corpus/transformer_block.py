# A pre-norm transformer block: vmapped LayerNorm/Linear over the sequence
# axis, self-attention via einsum, and residual connections.
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array


class TransformerBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    qkv: eqx.nn.Linear
    proj: eqx.nn.Linear
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, d_model, d_ff, key):
        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.norm2 = eqx.nn.LayerNorm(d_model)
        self.qkv = eqx.nn.Linear(d_model, d_model * 3, key=key)
        self.proj = eqx.nn.Linear(d_model, d_model, key=key)
        self.fc1 = eqx.nn.Linear(d_model, d_ff, key=key)
        self.fc2 = eqx.nn.Linear(d_ff, d_model, key=key)

    def __call__(self, x: Float[Array, "seq d_model"]):
        h = jax.vmap(self.norm1)(x)
        qkv = jax.vmap(self.qkv)(h)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        scores = jnp.einsum("qd,kd->qk", q, k)
        attn = jax.nn.softmax(scores, axis=-1)
        ctx = jnp.einsum("qk,kd->qd", attn, v)
        ctx = jax.vmap(self.proj)(ctx)
        x = x + ctx

        h = jax.vmap(self.norm2)(x)
        h = jax.nn.gelu(jax.vmap(self.fc1)(h))
        h = jax.vmap(self.fc2)(h)
        x = x + h
        return x
