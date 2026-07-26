# Adversarial-but-valid: `x`/`h` are reused as parameter/local names across
# a free function, two sibling methods, and a nested closure, each with a
# different shape — stress-testing per-function-scope isolation. Also
# chained method soup (reshape().transpose().sum()) and subscript
# gymnastics (mixed integer/slice/ellipsis/newaxis indexing in one
# expression). A locally-defined nested function called by name (`inner`,
# not a self.attr layer or an imported helper) is not traced — a genuine,
# error-free dark spot.
import jax.numpy as jnp
from jaxtyping import Float, Array


def standalone(x: Float[Array, "8 8"]):
    h = x.reshape(64).transpose(0).sum()
    return h


class ScopeSoup:
    def first(self, x: Float[Array, "4 6 8"]):
        h = x[0, :, 1:3]
        h = h[..., None]
        h = h.reshape(-1, 2).transpose(1, 0).sum(axis=1)
        corner = x[-1, ..., 0]
        return h, corner

    def second(self, x: Float[Array, "10 3"]):
        h = x.T
        h = h[:, ::2]

        def inner(h):
            y = h.reshape(-1).astype(jnp.float32)
            return y

        y = inner(h)
        h = h.sum(axis=0).reshape(1, -1)
        return h, y
