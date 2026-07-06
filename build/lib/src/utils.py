import os
import random
import librosa
import soundfile as sf
import numpy as np

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)

def random_sample(files, n = 5):
    return random.sample(files, n)