# A small convolutional classifier on a single (unbatched) image. Uses
# *direct* `self.layer(x)` application (not vmap), the dominant equinox/torch
# forward-method style.
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array


class ConvNet(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    fc: eqx.nn.Linear

    def __init__(self, key):
        self.conv1 = eqx.nn.Conv2d(3, 16, 3, key=key)
        self.conv2 = eqx.nn.Conv2d(16, 32, 3, key=key)
        self.fc = eqx.nn.Linear(32, 10, key=key)

    def __call__(self, x: Float[Array, "3 32 32"]):
        h = jax.nn.relu(self.conv1(x))
        h = jax.nn.relu(self.conv2(h))
        pooled = jnp.mean(h, axis=(1, 2))
        logits = self.fc(pooled)
        return logits
