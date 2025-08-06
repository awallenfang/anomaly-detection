import os
import torchaudio
import torchaudio.transforms as T
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from encoder import CNNEncoder
from decoder import CNNDecoder
from signal_gen import generate_dataset, prepare_for_pytorch, VibrationDataset, load_and_chunk_wav_with_overlap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_cnn(X, y, latent_dim = 64):
    # If encoder exists on disk, do no training
    if not os.path.isfile("./encoder.pth"):
        class CNNAutoencoder(nn.Module):
            def __init__(self, input_channels=1, latent_dim=latent_dim):
                super(CNNAutoencoder, self).__init__()
                self.encoder = CNNEncoder(input_channels, latent_dim)
                self.decoder = CNNDecoder(latent_dim)

            def forward(self, x):  # x: (batch, 1, 1000)
                z = self.encoder(x)
                x_recon = self.decoder(z)
                return x_recon
            
        # synthetic data


        train_dataset = VibrationDataset(X, y, only_normal=True)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


        model = CNNAutoencoder()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model = model.to(device)


        # Dummy training loop
        EPOCHS = 5
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0
            for batch_X, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                batch_X = batch_X.to(device)

                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_X)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}: avg reconstruction loss = {avg_loss:.6f}")
        encoder = model.encoder
        # Save encoder to disk
        torch.save(model.encoder, 'encoder.pth')
    else:
        encoder = torch.load('encoder.pth', weights_only=False)
    
    return encoder


class MFCCFeatures(nn.Module):
    def __init__(self, sample_rate=1000):
        super().__init__()
        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=40,
            melkwargs={
                'n_fft': 512,
                'hop_length': 64,
                'n_mels': 80
            }
        )

    def forward(self, X):
        mfcc_features = []
        for i in range(X.shape[0]):
            signal = X[i]  # shape: [1, signal_len]
            mfcc = self.mfcc_transform(signal)  # shape: [1, n_mfcc, time]
            mfcc = mfcc.squeeze(0)         # shape: [n_mfcc, time]
            mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
            mfcc_flat = mfcc.flatten()     # shape: [n_mfcc * time]
            mfcc_features.append(mfcc_flat)

        mfcc_features = torch.stack(mfcc_features)  # shape: [n_samples, n_features]
        return mfcc_features

