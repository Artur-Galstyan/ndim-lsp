# An eqx composite block, no nn.Sequential: equinox MultiheadAttention
# (num_heads-first ctor, distinct from torch's embed_dim-first order;
# tuple-unpacked output) feeding a vmapped LayerNorm and eqx.nn.MLP
# feedforward.
import jax
import equinox as eqx
from jaxtyping import Float, Array


class EqxAttnMlpBlock(eqx.Module):
    attn: eqx.nn.MultiheadAttention
    norm: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP

    def __init__(self, d_model, n_heads, key):
        self.attn = eqx.nn.MultiheadAttention(n_heads, d_model, key=key)
        self.norm = eqx.nn.LayerNorm(d_model)
        self.mlp = eqx.nn.MLP(d_model, d_model, width_size=d_model * 2, depth=2, key=key)

    def __call__(self, x: Float[Array, "seq d_model"]):
        attn_out, attn_weights = self.attn(x, x, x)
        h = x + attn_out
        normed = jax.vmap(self.norm)(h)
        out = jax.vmap(self.mlp)(normed)
        return out, attn_weights
