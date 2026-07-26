# A token classifier: embedding lookup, per-token logits, one_hot label
# encoding for a smoothing target, and take_along_axis to pull each row's
# top-prediction logit.
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Int, Array


class TokenClassifier(eqx.Module):
    embed: eqx.nn.Embedding
    proj: eqx.nn.Linear

    def __init__(self, vocab_size, d_model, n_classes, key):
        self.embed = eqx.nn.Embedding(vocab_size, d_model, key=key)
        self.proj = eqx.nn.Linear(d_model, n_classes, key=key)

    def __call__(
        self,
        tokens: Int[Array, "seq"],
        labels: Int[Array, "seq"],
        top_idx: Int[Array, "seq 1"],
    ):
        h = self.embed(tokens)
        logits = jax.vmap(self.proj)(h)

        one_hot_labels = jax.nn.one_hot(labels, 10)
        top_logit = jnp.take_along_axis(logits, top_idx, axis=-1)
        return logits, one_hot_labels, top_logit
