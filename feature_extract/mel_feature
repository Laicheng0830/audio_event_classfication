"""
!/usr/bin/env python
-*- coding:utf-8 -*-
Author: eric.lai
Created on 2019/5/10 14:42
"""

import librosa,h5py,os
from scipy.signal import spectrogram
import numpy as np
import preprocess_data.config as config
import soundfile


N_MEL = 12
N_WINDO = 128
N_OVERLAP = 64
TIME = 32
TIME_DELAY = 10

def extract_feature(data_array):
    features = []
    for i in range(data_array.shape[0]):
        features.append(extract_logmel(data_array[i]))
    return features

def read_audio(path, target_fs=None):
    """read audio"""
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs

def extract_logmel(audio,fs):
    # Mel filter bank
    melW = librosa.filters.mel(sr=fs,
                               n_fft=N_WINDO,
                               n_mels=N_MEL,
                               fmin=0.,
                               fmax=8000.)
    # Compute spectrogram
    ham_win = np.hamming(N_WINDO)
    [f, t, x] = spectrogram(
        x=audio,
        window=ham_win,
        nperseg=N_WINDO,
        noverlap=N_OVERLAP,
        detrend=False,
        return_onesided=True,
        mode='magnitude')
    x = x.T
    x = np.dot(x, melW.T)
    x = np.log(x + 1e-8)
    x = x.astype(np.float32)
    return x

def save_h5py(feature,label,filename):
    with h5py.File(filename, 'w') as f:
        subgroup = f.create_group('subgroup')
        subgroup.create_dataset('feature', data=feature)
        subgroup.create_dataset('label', data=label)


def read_h5py(filename):
    with h5py.File(filename, 'r') as f:
        # def prtname(name):
        #     print(name)
        # f.visit(prtname)
        subgroup = f['subgroup']
        data1 = subgroup['feature']
        data2 = subgroup['label']
        feature = data1.value
        label = data2.value
        print("data1 feature:", feature)
        print("data1 label:", label)

if __name__ == '__main__':
    # feature = [[1,2,3],[4,5,6]]
    # label = [1,2]
    # save_h5py(feature,label,'data.h5')
    # read_h5py('data.h5')
    DIR = config.config('pubg')
    feature = []
    label = []
    for i in range(len(DIR)):
        for root, dirs, files in os.walk(DIR[i]):
            for file in files:
                wave_file = DIR[i]+file
                wave_data, fs = read_audio(wave_file)
                feature_temp = extract_logmel(wave_data,fs)
                feature.append(feature_temp)
                label.append(i)
                print("runing:",wave_file)
    save_h5py(feature,label,'feature_test.h5')
    print(np.size(feature),np.size(label))

