# A tiny embedding-based language model head: token embedding lookup,
# positional embedding added in-place, projection to vocabulary logits.
# Exercises eqx.nn.Embedding, vmap over the sequence axis, and augmented
# assignment (`h += pos`).
import jax
import equinox as eqx
from jaxtyping import Float, Int, Array


class TinyLM(eqx.Module):
    embed: eqx.nn.Embedding
    proj: eqx.nn.Linear
    pos: Float[Array, "seq d_model"]

    def __init__(self, vocab_size, d_model, key):
        self.embed = eqx.nn.Embedding(vocab_size, d_model, key=key)
        self.proj = eqx.nn.Linear(d_model, vocab_size, key=key)

    def __call__(self, tokens: Int[Array, "seq"]):
        h = self.embed(tokens)
        h += self.pos
        normed = h / 2.0
        logits = jax.vmap(self.proj)(normed)
        return logits
