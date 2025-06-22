# Anomaly detection

Here is a model I came up with for a specific anomaly detection task. As input data it uses vibrations picked up from a board with motors attached.

For the training it solely requires data recorded by having the motors run. Then it can detect anomalies from that state, like a motor being disabled or similar.

It generally consists of two steps, a feature extractor and a one-class SVM. The feature extractor was implemented in two ways. 

A simple MFCC extractor that is no AI model, but rather a process of extracting frequency info. 

And a trained CNN model that takes the encoder from an autoencoder to get a latent feature vector.

Interestingly in tests the MFCC extractor workrd much better compared to the CNN approach. Thus it uses MFCC by default

## Usage

`train.py` is the script that trains and saves the models to files. 
The data for training should be audio files saved in training_audio/normal with a sample rate of 1000Hz

`main.py` runs the model and predicts the state based on the mic input coming from the default mic