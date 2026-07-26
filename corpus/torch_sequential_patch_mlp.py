# nn.Sequential chain including a nested Sequential and an einops
# Rearrange layer composed in the middle (LayerKind::Sequential threading
# through LayerKind::EinopsPattern and back into plain Linear/ReLU) — the
# now-supported Sequential/Rearrange composition, exercised end to end via
# `self.<attr>(x)` direct calls rather than a bare free-function chain.
import torch.nn as nn
from einops.layers.torch import Rearrange
from jaxtyping import Float, Array


class PatchMLP(nn.Module):
    def __init__(self, dim, n_classes):
        super().__init__()
        self.stem = nn.Sequential(
            Rearrange("c (h p1) (w p2) -> (h w) (p1 p2 c)", p1=16, p2=16),
            nn.Linear(16 * 16 * 3, dim),
            nn.ReLU(),
        )
        self.body = nn.Sequential(
            nn.Sequential(nn.Linear(dim, dim), nn.ReLU()),
            nn.Linear(dim, dim),
        )
        self.head = nn.Sequential(nn.Linear(dim, n_classes))

    def forward(self, img: Float[Array, "3 224 224"]):
        tokens = self.stem(img)
        pooled = tokens.mean(axis=0)
        hidden = self.body(pooled)
        logits = self.head(hidden)
        return logits
