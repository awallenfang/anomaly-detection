import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNDecoder(nn.Module):
    def __init__(self, latent_dim=64, output_length=1000):
        super(CNNDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 125 * 128)
        self.deconv1 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv3 = nn.ConvTranspose1d(32, 1, kernel_size=7, stride=2, padding=3, output_padding=1)

    def forward(self, z):  # z: (batch, latent_dim)
        x = self.fc(z)                  # (batch, 125 * 128)
        x = x.view(-1, 128, 125)        # (batch, 128, 125)
        x = F.relu(self.deconv1(x))     # (batch, 64, ~250)
        x = F.relu(self.deconv2(x))     # (batch, 32, ~500)
        x = self.deconv3(x)             # (batch, 1, ~1000)
        return x