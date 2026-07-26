# A deeper flax.linen CNN alternating avg_pool and max_pool between two
# conv blocks. flax_mlp.py already exercises Dense/Conv/avg_pool; max_pool
# shares the same KnownFunction::FlaxPool code path but has no dedicated
# corpus example yet. Also exercises a tuple-axis `jnp.mean` global pool
# feeding straight into Dense (rank-1 input, no reshape/flatten needed).
import jax.numpy as jnp
import flax.linen as nn
from jaxtyping import Float, Array


class FlaxConvPoolNet(nn.Module):
    @nn.compact
    def __call__(self, x: Float[Array, "64 64 3"]):
        h = nn.Conv(features=8, kernel_size=(3, 3))(x)
        h = nn.relu(h)
        h = nn.avg_pool(h, window_shape=(2, 2), strides=(2, 2))
        h = nn.Conv(features=16, kernel_size=(3, 3))(h)
        h = nn.gelu(h)
        h = nn.max_pool(h, window_shape=(2, 2), strides=(2, 2))
        pooled = jnp.mean(h, axis=(0, 1))
        hidden = nn.Dense(features=32)(pooled)
        hidden = nn.relu(hidden)
        logits = nn.Dense(features=10)(hidden)
        return logits
