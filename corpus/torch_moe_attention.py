# A torch attention + routing block exercising tensor methods: chunk
# (fused-QKV split), mT (key transpose), masked_fill (causal mask),
# softmax, gather (per-row expert logit lookup), and topk (routing gate).
import torch.nn as nn
from jaxtyping import Float, Array


class MaskedRoutedAttention(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.router = nn.Linear(d_model, n_experts)

    def forward(
        self,
        x: Float[Array, "seq d_model"],
        mask: Float[Array, "seq seq"],
        expert_idx: Float[Array, "seq 2"],
    ):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        scores = q @ k.mT
        masked = scores.masked_fill(mask, -1e9)
        weights = masked.softmax(dim=-1)
        ctx = weights @ v
        out = self.out_proj(ctx)

        route_logits = self.router(ctx)
        top_vals, top_idx = route_logits.topk(2, dim=-1)
        chosen = route_logits.gather(-1, expert_idx.long())
        return out, top_vals, chosen
