# ViT-style patch embedding using einops rearrange/reduce/repeat — the
# dominant idiom in modern vision code. einops is not modelled yet.
import jax.numpy as jnp
from einops import rearrange, reduce, repeat
from jaxtyping import Float, Array
import equinox as eqx


class PatchEmbed(eqx.Module):
    proj: eqx.nn.Linear

    def __init__(self, patch, dim, key):
        self.proj = eqx.nn.Linear(patch * patch * 3, dim, key=key)

    def __call__(self, img: Float[Array, "3 224 224"], cls: Float[Array, "dim"]):
        patches = rearrange(img, "c (h p1) (w p2) -> (h w) (p1 p2 c)", p1=16, p2=16)
        pooled = reduce(patches, "n d -> d", "mean")
        stacked = repeat(cls, "d -> n d", n=4)
        scaled = patches / 255.0
        return patches, pooled, stacked, scaled
