# A torch training-step-style file: a small classifier's `forward`, plus a
# sibling `loss` method that manually reimplements cross-entropy (NLL via
# log_softmax + gather, reduced to a scalar) and a one_hot encoding of the
# same integer targets. `self.forward(x)` called from a sibling method
# (not a self.attr layer) is not traced, so `logits` and everything
# downstream of it inside `loss` stays dark by design; `index` and
# `one_hot_targets` only depend on `targets` and stay shaped.
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Array


class Classifier(nn.Module):
    def __init__(self, in_features, hidden, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x: Float[Array, "batch in_features"]):
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)
        return logits

    def loss(
        self,
        x: Float[Array, "batch in_features"],
        targets: Float[Array, "batch"],
        n_classes: int,
    ):
        logits = self.forward(x)
        log_probs = F.log_softmax(logits, dim=-1)
        index = targets.long().unsqueeze(-1)
        picked = log_probs.gather(-1, index)
        nll = -picked.squeeze(-1)
        scalar_loss = nll.mean()
        one_hot_targets = F.one_hot(targets.long(), n_classes)
        return scalar_loss, one_hot_targets
