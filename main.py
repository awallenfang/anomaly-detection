import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from encoder import CNNEncoder
from decoder import CNNDecoder
from signal_gen import generate_dataset, prepare_for_pytorch, VibrationDataset
from tqdm import tqdm
class CNNAutoencoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=64):
        super(CNNAutoencoder, self).__init__()
        self.encoder = CNNEncoder(input_channels, latent_dim)
        self.decoder = CNNDecoder(latent_dim)

    def forward(self, x):  # x: (batch, 1, 1000)
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
    
# synthetic data

X, y= generate_dataset()
X = prepare_for_pytorch(X)

train_dataset = VibrationDataset(X, y, only_normal=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = CNNAutoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Dummy training loop
EPOCHS = 20
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

