import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from src.io import load_channel
from pathlib import Path

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
def plot_spectrogram(signal, sr, ax = None, title = None):
    d = librosa.amplitude_to_db(
        np.abs(librosa.stft(signal)),
        ref = np.max
    )
    if ax is None:
        fig, ax = plt.figure(figsize = (10, 4))
    img = librosa.display.specshow(
        d,
        sr = sr,
        x_axis = "time",
        y_log = "log",
        cmap = "magma",
        ax = ax
    )
    if title:
        ax.set_title(title)
    return img

# Spectrogram of a group
def plot_group_spectrograms(dataframe, folder, channel=0, title=""):
    fig, axes = plt.subplots(len(dataframe), 1, figsize=(12, 3*len(dataframe)), constrained_layout=True)
    if len(dataframe) == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=16)
    for ax, (_, row) in zip(axes, dataframe.iterrows()):
        file_path = Path(folder) / row["file_name"]
        signal, sr = load_channel(file_path, channel)
        img = plot_spectrogram(signal, sr, ax=ax, title=row["file_name"])
    fig.colorbar(img, ax=axes, format="%+2.0f dB")
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

