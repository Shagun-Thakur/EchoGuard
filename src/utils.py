import os
import random
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from src.config import *

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)

def random_sample(files, n = 5):
    return random.sample(files, n)

def get_audio_path(file_name, label):
    """
    Returns the full path of an audio file based on its label.
    """
    if label == "Normal":
        return normal_path/f"{file_name}"
    return abnormal_path/f"{file_name}"