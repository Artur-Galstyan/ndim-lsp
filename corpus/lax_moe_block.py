# A jax.lax-heavy block: dynamic_slice for a fixed local window,
# broadcast_in_dim for an explicit target-shape broadcast, lax.map to apply
# a linear layer over a leading axis without vmap's stricter batching, and
# lax.top_k for an expert-routing gate.
import jax
from jax import lax
import equinox as eqx
from jaxtyping import Float, Array


class LaxBlock(eqx.Module):
    proj: eqx.nn.Linear
    gate: eqx.nn.Linear

    def __init__(self, d_model, n_experts, key):
        self.proj = eqx.nn.Linear(d_model, d_model, key=key)
        self.gate = eqx.nn.Linear(d_model, n_experts, key=key)

    def __call__(
        self,
        x: Float[Array, "seq d_model"],
        bias: Float[Array, "d_model"],
    ):
        window = lax.dynamic_slice(x, (0, 0), (4, 16))
        broadcasted = lax.broadcast_in_dim(bias, (4, 16), (1,))
        combined = window + broadcasted

        mapped = lax.map(self.proj, combined)

        gate_logits = jax.vmap(self.gate)(mapped)
        top_vals, top_idx = lax.top_k(gate_logits, 2)
        return mapped, top_vals
