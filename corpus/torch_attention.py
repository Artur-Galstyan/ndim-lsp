# A torch decoder block: nn.MultiheadAttention (not classified yet),
# x.expand for a broadcasted mask, and in-place residual accumulation.
import torch
import torch.nn as nn
from jaxtyping import Float, Array


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(512, 8)
        self.norm = nn.LayerNorm(512)
        self.ff = nn.Linear(512, 512)

    def forward(
        self,
        x: Float[Array, "seq 512"],
        mask_row: Float[Array, "1 seq"],
    ):
        mask = mask_row.expand(8, -1)
        attn_out, attn_weights = self.attn(x, x, x)
        h = x + attn_out
        h = self.norm(h)
        h += self.ff(h)
        return h
