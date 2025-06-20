import numpy as np
import matplotlib.pyplot as plt
import random

# Sampling parameters
SAMPLE_RATE = 1000  # Hz
DURATION = 1.0      # seconds
N_SAMPLES = int(SAMPLE_RATE * DURATION)

def generate_base_signal(f_base=60, harmonics=[120, 180], noise_std=0.05, amplitude=1.0):
    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * f_base * t)
    for h in harmonics:
        signal += (amplitude / 2) * np.sin(2 * np.pi * h * t)
    noise = np.random.normal(0, noise_std, size=t.shape)
    return signal + noise

def generate_anomalous_signal(base_signal, kind='shift'):
    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
    modified = base_signal.copy()
    
    if kind == 'shift':
        modified *= 1.3  # amplitude spike
    elif kind == 'glitch':
        start = random.randint(200, 600)
        modified[start:start+50] += np.random.normal(0, 0.5, size=50)
    elif kind == 'drop':
        modified[int(0.4*N_SAMPLES):int(0.5*N_SAMPLES)] = 0
    elif kind == 'modulate':
        mod = 1 + 0.3 * np.sin(2 * np.pi * 5 * t)
        modified *= mod
    elif kind == 'frequency':
        modified = generate_base_signal(f_base=90)
    return modified

def generate_dataset(n_normal=1000, n_anomalous=100):
    X = []
    y = []

    for _ in range(n_normal):
        sig = generate_base_signal()
        X.append(sig)
        y.append(0)
    
    for _ in range(n_anomalous):
        base = generate_base_signal()
        kind = random.choice(['shift', 'glitch', 'drop', 'modulate', 'frequency'])
        sig = generate_anomalous_signal(base, kind=kind)
        X.append(sig)
        y.append(1)

    X = np.array(X)
    y = np.array(y)
    return X, y

def prepare_for_pytorch(X):
    # Normalize to zero-mean, unit-variance
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-6)
    # Reshape to (batch, 1, length)
    X = X[:, np.newaxis, :]
    return X.astype(np.float32)


import torch
from torch.utils.data import Dataset, DataLoader

class VibrationDataset(Dataset):
    def __init__(self, X, y, only_normal=False):
        if only_normal:
            mask = y == 0
            self.X = torch.tensor(X[mask], dtype=torch.float32)
            self.y = torch.tensor(y[mask], dtype=torch.long)
        else:
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]