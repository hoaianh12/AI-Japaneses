import librosa
import torch
import numpy as np
import config

def extract_mfcc(file_path):
    """ Trích xuất đặc trưng MFCC từ file âm thanh """
    y, sr = librosa.load(file_path, sr=config.SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.N_MFCC)
    return torch.tensor(mfcc).unsqueeze(0)
