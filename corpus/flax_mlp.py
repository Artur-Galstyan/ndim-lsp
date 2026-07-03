# A flax.linen MLP + conv stem. Flax layers are channels-LAST (opposite of
# the torch/equinox convention) and are not classified yet — expected dark
# spots that rank the flax gap.
import jax
import flax.linen as nn
from jaxtyping import Float, Array


class FlaxNet(nn.Module):
    @nn.compact
    def __call__(self, x: Float[Array, "32 32 3"]):
        h = nn.Conv(features=16, kernel_size=(3, 3))(x)
        h = nn.relu(h)
        h = nn.avg_pool(h, window_shape=(2, 2), strides=(2, 2))
        flat = h.reshape(-1)
        h2 = nn.Dense(features=64)(flat)
        logits = nn.Dense(features=10)(h2)
        return logits
