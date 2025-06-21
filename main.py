import sounddevice as sd
import torch
import torchaudio
import numpy as np
import joblib
from collections import deque
import time

from feature_extractors import MFCCFeatures
# Assuming your pipeline is named `pipeline`
pipeline = joblib.load("svm_pipeline.pkl")
print(pipeline)

mfcc_transform = MFCCFeatures()

# Configuration
SAMPLE_RATE = 1000
WINDOW_SIZE = 1000  # 1 second window
HOP_SIZE = 500      # 50% overlap

audio_buffer = deque(maxlen=WINDOW_SIZE)


def classify_chunk(chunk):
    # Convert to torch tensor
    waveform = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)  # [1, 1000]
    mfcc = mfcc_transform(waveform)       # [1, n_mfcc, time]
    mfcc_flat = mfcc.flatten().numpy()    # [n_features]
    
    # Predict
    pred = pipeline.predict([mfcc_flat])[0]
    score = pipeline.decision_function([mfcc_flat])
    if score <= -1:
        pred = 1
    else:
        pred = 0
    return pred

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_buffer.extend(indata[:, 0])
    
    # Process when we have enough samples
    if len(audio_buffer) == WINDOW_SIZE:
        chunk = np.array(audio_buffer)
        chunk = (chunk - chunk.mean()) / (chunk.std() + 1e-5)
        label = classify_chunk(chunk)
        print(f"Prediction: {'ANOMALY' if label == 1 else 'NORMAL'}")

        # Retain overlap
        retained = list(audio_buffer)[HOP_SIZE:]
        audio_buffer.clear()
        audio_buffer.extend(retained)


print("Listening...")
with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=audio_callback, blocksize=HOP_SIZE):
    while True:
        time.sleep(0.1)  # Keeps main thread alive