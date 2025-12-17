import librosa
import numpy as np
from scipy.spatial.distance import cosine

def extract_mfcc(path):
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)


def compare_audio(file1, file2):
    mfcc1 = extract_mfcc(file1)
    mfcc2 = extract_mfcc(file2)

    similarity = 1 - cosine(mfcc1, mfcc2)
    return similarity
