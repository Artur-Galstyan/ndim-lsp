# An equinox model combining an MLP encoder with an LSTMCell/GRUCell pair.
# Single-step direct self-attr calls resolve normally (approximated as just
# the hidden output, per the existing RnnCell rule); a jax.lax.scan over
# the same LSTMCell with a `(h, c)` tuple carry is opaque to the analyzer
# (unlike rnn_scan.py's bare-array GRU carry) — `carry_final`/`h_final`/
# `c_final` stay dark by design, a genuine error-free gap.
import jax
import equinox as eqx
from jaxtyping import Float, Array


class LSTMEncoder(eqx.Module):
    encoder: eqx.nn.MLP
    lstm: eqx.nn.LSTMCell
    gru: eqx.nn.GRUCell
    readout: eqx.nn.Linear

    def __init__(self, in_size, hidden, key):
        self.encoder = eqx.nn.MLP(in_size, hidden, width_size=hidden, depth=2, key=key)
        self.lstm = eqx.nn.LSTMCell(hidden, hidden, key=key)
        self.gru = eqx.nn.GRUCell(hidden, hidden, key=key)
        self.readout = eqx.nn.Linear(hidden, in_size, key=key)

    def __call__(
        self,
        xs: Float[Array, "seq in_size"],
        h0: Float[Array, "hidden"],
        c0: Float[Array, "hidden"],
    ):
        encoded = jax.vmap(self.encoder)(xs)
        step_h = self.lstm(encoded[0], (h0, c0))
        step_h2 = self.gru(step_h, step_h)

        def body(carry, x):
            h, c = carry
            h_new = self.lstm(x, (h, c))
            return (h_new, c), h_new

        carry_final, hs = jax.lax.scan(body, (h0, c0), encoded)
        h_final, c_final = carry_final
        out = self.readout(step_h2)
        return out, hs, h_final, c_final
