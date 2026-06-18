# A minimal multi-head self-attention block. Exercises matmul/einsum,
# reshape/transpose, softmax (shape-preserving), and slicing.
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array


class SelfAttention(eqx.Module):
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    n_heads: int

    def __init__(self, d_model, n_heads, key):
        self.q_proj = eqx.nn.Linear(d_model, d_model, key=key)
        self.k_proj = eqx.nn.Linear(d_model, d_model, key=key)
        self.v_proj = eqx.nn.Linear(d_model, d_model, key=key)
        self.out_proj = eqx.nn.Linear(d_model, d_model, key=key)
        self.n_heads = n_heads

    def __call__(self, x: Float[Array, "seq d_model"]):
        q = jax.vmap(self.q_proj)(x)
        k = jax.vmap(self.k_proj)(x)
        v = jax.vmap(self.v_proj)(x)

        scores = jnp.einsum("qd,kd->qk", q, k)
        weights = jax.nn.softmax(scores, axis=-1)
        context = jnp.einsum("qk,kd->qd", weights, v)

        first_token = context[0]
        out = jax.vmap(self.out_proj)(context)
        return out
