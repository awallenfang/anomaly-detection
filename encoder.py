import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np

class CNNEncoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=64, sample_rate = 1000):
        super(CNNEncoder, self).__init__()
        self.mfcc_transform = T.MFCC(
            sample_rate=1000,
            n_mfcc=40,
            melkwargs={
                'n_fft': 256,
                'hop_length': 128,
                'n_mels': 64
            }
        )

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.fc = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = self.mfcc_transform(x)
        x = self.conv(x)
        x = x.squeeze(-1)
        return self.fc(x)

