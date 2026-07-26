# ViT-style transformer block: Conv2d patch embedding, an einops rearrange
# using a `...` ellipsis (the pattern also works if a leading batch axis is
# vmapped in by the caller), einops.einsum for attention, and a pre-norm
# residual MLP. Complements corpus/einops_vit.py's non-ellipsis rearrange
# with the ellipsis + einops.einsum path.
import jax
import equinox as eqx
from einops import rearrange
from einops import einsum as eeinsum
from jaxtyping import Float, Array


class ViTBlock(eqx.Module):
    patch_proj: eqx.nn.Conv2d
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, dim, key):
        self.patch_proj = eqx.nn.Conv2d(3, dim, 16, stride=16, key=key)
        self.norm1 = eqx.nn.LayerNorm(dim)
        self.norm2 = eqx.nn.LayerNorm(dim)
        self.q_proj = eqx.nn.Linear(dim, dim, key=key)
        self.k_proj = eqx.nn.Linear(dim, dim, key=key)
        self.v_proj = eqx.nn.Linear(dim, dim, key=key)
        self.out_proj = eqx.nn.Linear(dim, dim, key=key)
        self.fc1 = eqx.nn.Linear(dim, dim * 4, key=key)
        self.fc2 = eqx.nn.Linear(dim * 4, dim, key=key)

    def __call__(self, img: Float[Array, "3 224 224"]):
        patches = self.patch_proj(img)
        tokens = rearrange(patches, "... c h w -> ... (h w) c")

        normed = jax.vmap(self.norm1)(tokens)
        q = jax.vmap(self.q_proj)(normed)
        k = jax.vmap(self.k_proj)(normed)
        v = jax.vmap(self.v_proj)(normed)

        scores = eeinsum(q, k, "q d, k d -> q k")
        weights = jax.nn.softmax(scores, axis=-1)
        ctx = eeinsum(weights, v, "q k, k d -> q d")
        attn_out = jax.vmap(self.out_proj)(ctx)
        x = tokens + attn_out

        h = jax.vmap(self.norm2)(x)
        h = jax.nn.gelu(jax.vmap(self.fc1)(h))
        h = jax.vmap(self.fc2)(h)
        return x + h
