import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from src.io import load_channel

# Waveform
def plot_waveforms(files, channel = 0, title = None):
    fig, axes = plt.subplots(
        len(files),
        1,
        figsize=(12, 10),
        sharex = True
    )
    if title:
        fig.suptitle(title)
    for ax, file in zip(axes, files):
        signal, sr = load_channel(
            file,
            channel
        )
        librosa.display.waveshow(signal, sr = sr, ax = ax)
        ax.set_title(file.name)
    plt.tight_layout()
    plt.show()

# Spectrogram
def plot_spectrogram(signal, sr):
    d = librosa.amplitude_to_db(
        np.abs(librosa.stft(signal)),
        ref = np.max
    )
    plt.figure(figsize = (10, 4))
    librosa.display.specshow(
        d,
        sr = sr,
        x_axis = "time",
        y_log = "log"
    )
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# Histogram
def plot_histogram(data,
                   bins = 30,
                   xlabel = "",
                   ylabel = "Frequency",
                   title = ""):
    plt.figure(figsize = (8, 5))
    plt.hist(data, bins = bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_boxplot(data, labels, ylabel):
    plt.figure(figsize = (8, 5))
    plt.boxplot(data, labels = labels)
    plt.ylabel(ylabel)
    plt.show()

