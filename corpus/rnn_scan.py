# A GRU-style recurrent cell driven by `jax.lax.scan`, the canonical JAX RNN
# pattern (separate input/hidden projections). `scan` and the `self.step`
# method call are not modelled yet — expected dark spots that keep those gaps
# ranked in the coverage report.
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array


class GRUCell(eqx.Module):
    wz_x: eqx.nn.Linear
    wz_h: eqx.nn.Linear
    wh_x: eqx.nn.Linear
    wh_h: eqx.nn.Linear

    def __init__(self, input_size, hidden_size, key):
        self.wz_x = eqx.nn.Linear(input_size, hidden_size, key=key)
        self.wz_h = eqx.nn.Linear(hidden_size, hidden_size, key=key)
        self.wh_x = eqx.nn.Linear(input_size, hidden_size, key=key)
        self.wh_h = eqx.nn.Linear(hidden_size, hidden_size, key=key)

    def step(
        self,
        h: Float[Array, "hidden_size"],
        x: Float[Array, "input_size"],
    ) -> Float[Array, "hidden_size"]:
        zx = self.wz_x(x)
        zh = self.wz_h(h)
        z = jax.nn.sigmoid(zx + zh)
        hx = self.wh_x(x)
        hh = self.wh_h(h)
        h_tilde = jnp.tanh(hx + hh)
        h_new = (1 - z) * h + z * h_tilde
        return h_new

    def __call__(
        self,
        xs: Float[Array, "seq input_size"],
        h0: Float[Array, "hidden_size"],
    ):
        def body(carry, x):
            new_carry = self.step(carry, x)
            return new_carry, new_carry

        h_final, hs = jax.lax.scan(body, h0, xs)
        return h_final, hs
