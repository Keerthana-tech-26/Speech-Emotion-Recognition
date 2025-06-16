import librosa
import numpy as np

def extract_features(file):
    audio, sr = librosa.load(file, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)
