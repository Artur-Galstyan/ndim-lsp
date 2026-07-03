# PCA / least-squares style linear algebra: multi-output tuple unpacking for
# svd/qr/eigh and meshgrid. Only `split` unpacking is modelled so far.
import jax.numpy as jnp
from jaxtyping import Float, Array


def pca(x: Float[Array, "n d"], w: Float[Array, "d d"]):
    centered = x - jnp.mean(x, axis=0)
    u, s, vt = jnp.linalg.svd(centered)
    q, r = jnp.linalg.qr(centered)
    evals, evecs = jnp.linalg.eigh(w)
    projected = centered @ w
    return projected


def grid_coords(xs: Float[Array, "nx"], ys: Float[Array, "ny"]):
    gx, gy = jnp.meshgrid(xs, ys)
    return gx, gy
