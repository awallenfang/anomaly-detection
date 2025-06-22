import joblib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from signal_gen import prepare_for_pytorch, VibrationDataset, load_and_chunk_wav_with_overlap
from feature_extractors import train_cnn, MFCCFeatures

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Generating dataset")
file_and_label = []

normal_names = os.listdir("training_audio/normal")
for name in normal_names:
    if name.endswith(".flac"):
        file_and_label.append(("training_audio/normal/"+name, 0))

anomalous_names = os.listdir("training_audio/anomaly")
for name in anomalous_names:
    if name.endswith(".flac"):
        file_and_label.append(("training_audio/anomaly/"+name, 1))

X = None
y = None
for file, label in file_and_label:
    
    X_temp,y_temp = load_and_chunk_wav_with_overlap(file, label)
    if X is None:
        X = X_temp
        y = y_temp
    else:
        X = np.concat([X,X_temp])
        y = np.concat([y,y_temp])

print("Preparing dataset")
X = prepare_for_pytorch(X)

method = "mfcc"
if method == "cnn":
    print("Using CNN")
    encoder = train_cnn(X, y)
else:
    print("Using MFCC")
    encoder = MFCCFeatures()

print("Fetched encoder")

# Extract encode and calculate features for all input data

train_dataset = VibrationDataset(X, y, only_normal=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print("Encoding training features")
features = []
labels = []
with torch.no_grad():
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        latent = encoder(batch_X)  # Shape: (B, C, T)
        features.append(latent.cpu())
        labels.append(batch_y)

train_features = torch.cat(features)
train_labels = torch.cat(labels)

train_features = train_features.numpy()
train_labels = train_labels.numpy()

test_dataset = VibrationDataset(X, y, only_normal=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


print("Encoding testing features")
features = []
labels = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        latent = encoder(batch_X)  # Shape: (B, C, T)
        flat = latent.view(latent.size(0), -1)  # Flatten to 2D: (B, C*T)
        features.append(flat.cpu())
        labels.append(batch_y)

test_features = torch.cat(features)
test_labels = torch.cat(labels)

test_features = test_features.numpy()
test_labels = test_labels.numpy()

from sklearn.pipeline import make_pipeline

print("Training SVM")
# Normalize features
pipeline = make_pipeline(StandardScaler(), OneClassSVM(nu=0.02, kernel="rbf", gamma='scale', verbose=True))
pipeline.fit(train_features)


# Assuming your pipeline is named `pipeline`
joblib.dump(pipeline, "svm_pipeline.pkl")

pred = pipeline.predict(test_features)
unique, counts = np.unique(pred, return_counts=True)
# One-Class SVM outputs: 1 (inlier), -1 (outlier)
# You may need to map: -1 → 1 (anomaly), 1 → 0 (normal)
if np.min(pred) == -1:
    pred = (pred == -1).astype(int)  # 1 = anomaly

from sklearn.metrics import classification_report
print(classification_report(test_labels, pred))
