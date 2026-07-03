# A torch-style CNN with explicit pooling layers (MaxPool2d halves the
# spatial dims, AdaptiveAvgPool2d collapses them to output_size).
import torch
import torch.nn as nn
from jaxtyping import Float, Array


class PoolNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x: Float[Array, "3 32 32"]):
        h = self.conv1(x)
        h = self.pool1(h)
        h = self.conv2(h)
        h = self.pool2(h)
        pooled = h.mean(axis=(1, 2))
        logits = self.fc(pooled)
        return logits
