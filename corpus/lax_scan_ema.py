# A jax.lax.scan pipeline distinct from rnn_scan.py's bare-array GRU
# carry: a running per-feature EMA of (mean, var) statistics threaded as a
# plain tuple carry (not a dict, and not a single array). `x_abs`/
# `xs_scaled` show the plain jaxtyping-annotated params are shaped
# normally; the analyzer only models a scan whose `init` argument is
# itself a simple named array (`final_carry` gets `init`'s shape), so a
# tuple-literal `init` and the resulting tuple carry are un-shaped by
# design here — genuine, error-free dark spots isolated to the carry.
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array


def ema_step(carry, x: Float[Array, "features"], decay: float = 0.9):
    x_abs = jnp.abs(x)
    mean, var = carry
    delta = x - mean
    new_mean = mean + (1 - decay) * delta
    new_var = decay * var + (1 - decay) * delta * delta
    return (new_mean, new_var), new_mean


def running_ema(
    xs: Float[Array, "seq features"],
    mean0: Float[Array, "features"],
    var0: Float[Array, "features"],
):
    xs_scaled = xs * 2.0
    init = (mean0, var0)
    final_carry, means = jax.lax.scan(ema_step, init, xs)
    final_mean, final_var = final_carry
    scaled = means * 2.0
    return final_mean, final_var, scaled, xs_scaled
