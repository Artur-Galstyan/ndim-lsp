# A Mamba-style selective state-space block. Realistic shape-tracing target:
# exercises self-attribute layers, vmap, split, x.shape unpacking, chained
# methods, and (currently unsupported) slicing.
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array


class SelectiveStateSpace(eqx.Module):
    input_proj: eqx.nn.Linear
    delta_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    A_log: Float[Array, "d_inner d_state"]
    D: Float[Array, "d_inner"]
    dt_rank: int
    d_inner: int
    d_state: int

    def __init__(self, d_inner, d_state, dt_rank, key):
        self.input_proj = eqx.nn.Linear(d_inner, dt_rank + d_state * 2, key=key)
        self.delta_proj = eqx.nn.Linear(dt_rank, d_inner, key=key)
        self.out_proj = eqx.nn.Linear(d_inner, d_inner, key=key)
        self.dt_rank = dt_rank
        self.d_inner = d_inner
        self.d_state = d_state

    def __call__(self, x: Float[Array, "seq_length d_inner"]):
        L, _ = x.shape

        A = -jnp.exp(self.A_log.astype(jnp.float32))
        D = self.D.astype(jnp.float32)

        delta_b_c = jax.vmap(self.input_proj)(x)
        delta, B, C = jnp.split(
            delta_b_c,
            [self.dt_rank, self.dt_rank + self.d_state],
            axis=-1,
        )
        delta = jax.nn.softplus(jax.vmap(self.delta_proj)(delta))

        first_step = x[0]
        prefix = x[:, : self.d_inner]
        delta_last = delta[..., None]

        y = jax.vmap(self.out_proj)(x)
        skip = y + x
        return skip
