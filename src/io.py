
import librosa
import soundfile as sf

def load_audio(path, sr = None, mono = False):
    signal, sr = librosa.load(
        path,
        sr = sr,
        mono = mono
    )
    return signal, sr

def load_channel(path, channel = 0):
    signal, sr = librosa.load(
        path,
        sr = None,
        mono = False
    )
    return signal[channel], sr

def audio_info(path):
    return sf.info(path)
