import os
import random
import librosa
import soundfile as sf
import numpy as np

def get_audio_files(folder_path, extension = ".wav"):
    """
    Return sorted list of audio file paths.
    """
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(extension)
    ]
    return sorted(files)

def load_audio(path, sr = None, mono = False):
    """
    Load complete multichannel audio.
    """
    signal, sr = librosa.load(
        path,
        sr = sr,
        mono = mono
    )
    return signal, sr

def load_channel(path, channel = 0):
    """
    Load one channel from multichannel recording.
    """
    signal, sr = librosa.load(
        path,
        sr = None,
        mono = False
    )
    return signal[channel], sr

def random_files(file_list, n = 5, seed = 42):
    random.seed(seed)
    return random.sample(file_list, n)

def audio_info(path):
    """
    Return SounFile information. 
    """
    return sf.info(path)

def duration(path):
    """
    Return duration of audio file in seconds.
    """
    return sf.info(path).duration

def sample_rate(path):
    """
    Return sample rate of audio file in Hz.
    """
    return sf.info(path).samplerate

def channels(path):
    """
    Return number of channels of audio file.
    """
    return sf.info(path).channels
