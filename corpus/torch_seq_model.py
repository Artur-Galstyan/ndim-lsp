# A torch sequence model exercising nn.LSTM (single-LHS: the analyzer
# approximates the primary output tensor only, not the (h_n, c_n) tuple)
# feeding a manual nn.GRUCell decoder loop.
import torch.nn as nn
from jaxtyping import Float, Array


class SeqModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size)
        self.cell = nn.GRUCell(hidden_size, hidden_size)
        self.readout = nn.Linear(hidden_size, input_size)

    def forward(
        self,
        x: Float[Array, "seq input_size"],
        h0: Float[Array, "hidden_size"],
    ):
        encoded = self.encoder(x)
        last = encoded[-1]
        h1 = self.cell(last, h0)
        h2 = self.cell(last, h1)
        out = self.readout(h2)
        return out
