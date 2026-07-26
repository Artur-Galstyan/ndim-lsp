# A torch upsampling decoder chaining three different upsampling idioms:
# ConvTranspose2d (learned upsampling), nn.Upsample (fixed-scale resize),
# and PixelShuffle (sub-pixel convolution head).
import torch.nn as nn
from jaxtyping import Float, Array


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(32, 32, 3, padding=1)
        self.shuffle_conv = nn.Conv2d(32, 128, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, z: Float[Array, "128"]):
        h = self.fc(z)
        h = h.reshape(64, 4, 4)
        h = self.deconv1(h)
        h = self.upsample(h)
        h = self.conv(h)
        h = self.shuffle_conv(h)
        out = self.shuffle(h)
        return out
