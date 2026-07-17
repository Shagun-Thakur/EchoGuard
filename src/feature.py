import librosa
import numpy as np
from src.io import *
import sys

# 1. RMS
def rms(signal):
    return librosa.feature.rms(y = signal)[0]

def mean_rms(signal):
    return np.mean(rms(signal))

# 2. Spectral Centroid
def spectral_centroid(signal, sr):
    return librosa.feature.spectral_centroid(
        y = signal,
        sr = sr
    )[0]

# 3. Spectral Bandwidth
def spectral_bandwidth(signal, sr):
    return librosa.feature.spectral_bandwidth(
        y = signal,
        sr = sr
    )[0]

# 4. MFCC
def mfcc(signal, sr, n_mfcc = 13):
    return librosa.feature.mfcc(
        y = signal,
        sr = sr,
        n_mfcc = n_mfcc
    )

# 5. Zero Crossing Rate
def zero_crossing_rate(signal):
    return librosa.feature.zero_crossing_rate(signal)[0]

# 6. Statistics
def statistics(feature):
    return{
        "mean": np.mean(feature),
        "std": np.std(feature),
        "meadian": np.median(feature),
        "min": np.min(feature),
        "max": np.max(feature),
        "q25": np.percentile(feature, 25),
        "q75": np.percentile(feature, 75),
        "iqr": np.percentile(feature, 75) - np.percentile(feature, 25),
        "delta": np.max(feature) - np.min(feature)
    }

# 7. Spectral Centroid Statistics
def spectral_centroid_statistics(signal, sr):
    centroid = librosa.feature.spectral_centroid(y = signal, sr = sr)[0]
    q25 = np.percentile(centroid, 25)
    q75 = np.percentile(centroid, 75)
    return{
    "centroid_mean": np.mean(centroid),
    "centroid_std": np.std(centroid),
    "centroid_min": np.min(centroid),
    "centroid_max": np.max(centroid),
    "centroid_delta":np.max(centroid) - np.min(centroid),
    "centroid_q25": q25,
    "centroid_q75": q75,
    "centroid_iqr": q75 - q25
    }

# MFCC Feature Extraction Function
def mfcc_statistics(signal, sr, n_mfcc = 20):
    mfcc = librosa.feature.mfcc(y = signal, sr = sr, n_mfcc = n_mfcc)
    features = {}
    for i in range(n_mfcc):
        features[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i+1}_std"] = np.std(mfcc[i])
    return features

# Compute Log-Mel Spectrogram
def compute_log_mel(signal):
    """ 
    Compute log-Mel spectrogram.
    """
    mel = librosa.feature.melspectrogram(
        y = signal,
        sr = 16000,
        n_fft = 1024,
        hop_length = 512,
        n_mels = 64,
        power = 2.0
    )
    log_mel = 20.0 / 2.0 * np.log10(mel + sys.float_info.epsilon)
    return log_mel

# Create 320-Dimensional Feature Vectors
def create_feature_vectors(log_mel):
    """
    Convert a log-Mel spectrogram into
    320-dimensional feature vectors by concatenating
    five consecutive frames.
    """
    frames = log_mel.T
    context = 5
    feature_vectors = []
    for i in range(len(frames) - context + 1):
        window = frames[i:i + context]
        feature_vectors.append(window.flatten())
    feature_vectors = np.asarray(feature_vectors)
    return feature_vectors

# Complete feature extraction pipeline
def extract_feature(filepath):
    signal, x = load_audio(filepath, sr = 16000, mono = True)   # x = sample rate
    log_mel = compute_log_mel(signal)
    vectors = create_feature_vectors(log_mel)
    return signal, log_mel, vectors
    