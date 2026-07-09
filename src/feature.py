import librosa
import numpy as np

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